#!/usr/bin/env python3

import os
import yaml
import rospy
import sys
import numpy as np
import tf
import cv2 


from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from rosgraph.names import REMAP
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from rospy import Subscriber, Publisher
from tf import TransformBroadcaster

from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from dt_apriltags import Detector

# local import 
from utils.rectification import Rectify
from utils.wheel_odometry import WheelOdometry
LEFT = 0
RIGHT = 2
FORWARD = 1
BACKWARD = -1

def calc_dist(ticks, resolution, radius):
    x = 2*np.pi*radius*float(ticks)/float(resolution)
    return x

def homography2transformation(H, K):
    # @H: homography, 3x3 matrix 
    # @K: intrinsic matrix, 3x3 matrix 
    # apply inv(K)
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    K_inv = np.array(
        [[1/fx,  0,      -cx/fx],
        [0,     1/fy,   -cy/fy],
        [0,     0,      1     ]]
    ) 
    Rt = K_inv @ H

    # normalize Rt matrix so that R columns have unit length 
    norm_1 = np.linalg.norm(Rt[:, 0])
    norm_2 = np.linalg.norm(Rt[:, 1])
    norm = np.sqrt(norm_1 * norm_2) # get the average norm of first two columns
    scale = 1/norm
    # WARNING: baselink is under the camera, which requires t_y to be positive  
    if Rt[1, 2] < 0:
        scale = -1 * scale
    
    norm_Rt = scale * Rt

    r1 = norm_Rt[:, 0]; r2 = norm_Rt[:, 1]
    r3 = np.cross(r1, r2)
    R = np.stack((r1, r2, r3), axis=1)
    
    # no idea why, this SVD will ruin the whole transformation!!! 
    # print("R (before polar decomposition):\n",R,"\ndet(R): ", np.linalg.det(R))
    u, s, vh = np.linalg.svd(R)
    R = u@vh
    # print("R (after polar decomposition):\n", R, "\ndet(R): ", np.linalg.det(R))

    T = np.zeros((4,4))
    T[0:3, 0:3] = R
    T[0:3, 3] = norm_Rt[:, 2]
    T[3,3] = 1.0

    return T


class EncoderLocNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

################################ params, variables and flags ####################################
        # Initialize the DTROS parent class
        super(EncoderLocNode, self).__init__(node_name, NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")
        self.node_name = rospy.get_name().strip("/")
        # Get static parameters
        self._radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius')
        self._baseline = rospy.get_param(f'/{self.veh_name}/kinematics_node/baseline')
        
        # load ground homography
        self.groun_homography = self.load_extrinsics()
        self.homography_g2p = np.linalg.inv(np.array(self.groun_homography).reshape((3,3)))

        self.at_camera_params = None
        self.at_tag_size = 0.065 # fixed param (real value)
        # self.at_tag_size = 2 # to scale pose matrix to homography
        
        # flag of whether camera info received or not
        self.camera_info_received = False
        # flag of whether initial localization finished  
        self.first_loc = False
        
        # TFs try rotate x-axis by 90 deg or -90 deg
        self.tf_mapFapriltag = np.array([   # from apriltag to map
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.09], # 9cm
            [0.0, 0.0, 0.0, 1.0]
        ]) 
        # self.tf_apriltag2map = np.array([   # from apriltag to map
        #     [1.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, -1.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.09], # 9cm
        #     [0.0, 0.0, 0.0, 1.0]
        # ]) 
        self.tf_cameraFapriltag = None # from apriltag to camera
        self.tf_mapFcamera = None # from camera to map 
        self.tf_cameraFbaselink = None # from baselink to camera

        self.tf_mapFbaselink = None # target tf, from baselink to map 

############################# member objects needed to init before pub&sub ##################
        self.odm = None 
        self.bridge = CvBridge()
        self.rectifier = None
        self.tf_bcaster = TransformBroadcaster()

        # apriltag detector
        self.at_detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
            #searchpath=[]
        )

############################## subscribers and publishers ####################################
        self.sub_encoder_ticks_left = rospy.Subscriber(
            f'/{self.veh_name}/left_wheel_encoder_node/tick',
            WheelEncoderStamped,
            self.cb_encoder_data_left,
            queue_size=1   
        )
        self.log(f"listening to {f'/{self.veh_name}/left_wheel_encoder_node/tick'}")
        
        self.sub_encoder_ticks_right = rospy.Subscriber(
            f'/{self.veh_name}/right_wheel_encoder_node/tick',
            WheelEncoderStamped,
            self.cb_encoder_data_right,
            queue_size=1
        )
        self.log(f"listening to {f'/{self.veh_name}/right_wheel_encoder_node/tick'}")
        
        ### In the lastest dt-car-interface, direction has already been considered
        ### Thus direction no longer needed to be considered 
        # self.sub_executed_commands = rospy.Subscriber(
        #     f'/{self.veh_name}/wheels_driver_node/wheels_cmd_executed',
        #     WheelsCmdStamped,
        #     self.cb_executed_commands,
        #     queue_size=1
        # )
        self.log(f"listening to {f'/{self.veh_name}/wheels_driver_node/wheels_cmd_executed'}")
        
        self.sub_camera_info = Subscriber(
            f'/{self.veh_name}/camera_node/camera_info', 
            CameraInfo,
            self.cb_camera_info, 
            queue_size=1
        )
        self.log(f"Subcribing to topic {f'/{self.veh_name}/camera_node/camera_info'}")


        self.sub_compressed_image = rospy.Subscriber(
            f'/{self.veh_name}/camera_node/image/compressed',
            CompressedImage,
            self.cb_compressed_image,
            queue_size=1
        )
        self.log(f"listening to {f'/{self.veh_name}/camera_node/image/compressed'}")

        # Publishers

        # self.pub_baselink = rospy.Publisher(
        #     "~TF/encoder_baselink",
        #     Float32,
        #     queue_size=1
        # )
        # self.log(f"Publishing data to {f'/{self.veh_name}/{self.node_name}/TF/encoder_baselink'}")

        # self.pub_integrated_distance_left = rospy.Publisher(
        #     "~left_wheel_distance",
        #     Float32,
        #     queue_size=1
        # )
        # self.log(f"Publishing data to {f'/{self.veh_name}/{self.node_name}/left_wheel_distance'}")

        # self.pub_integrated_distance_right = rospy.Publisher(
        #     "~right_wheel_distance",
        #     Float32,
        #     queue_size=1
        # )
        # self.log(f"Publishing data to {f'/{self.veh_name}/{self.node_name}/left_wheel_distance'}")

        self.log("Class EncoderLocNode initialized")
    
    
    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log("Can't find calibration file: %s.\n Using default calibration instead."
                     % cali_file, 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file,'r') as stream:
                calib_data = yaml.load(stream)
        except yaml.YAMLError:
            msg = 'Error in parsing calibration file %s ... aborting' % cali_file
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        return calib_data['homography']

    # broadcast a transform matrix as tf type 
    def broadcast_tf(self, tf_mat, time, # rospy.Time()
        child="encoder_baselink",
        parent="map"):


        def _matrix_to_quaternion(r):
            T = np.array((
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 1)
            ), dtype=np.float64)
            T[0:3, 0:3] = r
            return tf.transformations.quaternion_from_matrix(T)

        rvec = _matrix_to_quaternion(tf_mat[:3,:3])
        tvec = tf_mat[:3, 3].reshape(-1)

        self.tf_bcaster.sendTransform(
                                tvec.tolist(),
                                rvec.tolist(),
                                time,
                                child,
                                parent)
        return 

    def detect(self, img):

        greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(greyscale_img, True,
                            self.at_camera_params, self.at_tag_size)
        
        
        # calculate poses of all tags
        count = 0
        for tag in tags: # not None
            tf_cameraFapriltag = np.concatenate(
                (
                    np.concatenate((tag.pose_R, tag.pose_t),axis=1),  # 3x4
                    np.array([[0.0,0.0,0.0,1.0]]) # 1x4
                ), 
                axis=0
            )
            # select the position of first tag as map frame 
            if self.first_loc == False:

                self.tf_cameraFapriltag = tf_cameraFapriltag
                # calculate transform from baselink to map
                # T_c2m = T_c2a @ T_a2m
                self.tf_mapFcamera = self.tf_mapFapriltag @ np.linalg.inv(self.tf_cameraFapriltag) 
                # T_b2m = T_b2c @ T_c2m
                self.tf_mapFbaselink = self.tf_mapFcamera @ self.tf_cameraFbaselink
                # first localization finished 
                self.odm = WheelOdometry(self._radius, self._baseline, self.tf_mapFbaselink, self, frequency=10)
                
                # self.odm.start()
                
                self.first_loc = True   
                self.log("First localization finished!")
            # tf_mapFapriltag = self.tf_mapFcamera @ tf_cameraFapriltag

            # publish apriltag TF 
            # self.broadcast_tf(tf_april2map, rospy.Time.now(),
            #                 child=f"apriltag_{tag.tag_id}",
            #                 parent="map")

            # self.logdebug(f"publish tf_april2map, which is {tf_mapFapriltag}")
            
            count += 1
        
        # if count == 0:
        #     self.loginfo("No tag detected in the camera range.")
        
        return 

    def cb_encoder_data_left(self, msg):
        if self.first_loc:
            self.odm.update_wheel("left_wheel", msg)
        return 
        

    def cb_encoder_data_right(self, msg):
        # self.logdebug(f"Main: cb_encoder_data_right called")
        if self.first_loc:
            self.odm.update_wheel("right_wheel", msg)
        pass

    ### In the lastest dt-car-interface, direction has already been considered
    ### Thus direction no longer needed to be considered 
    # def cb_executed_commands(self, msg):
    #     """ Use the executed commands to determine the direction of travel of each wheel.
    #     """
    #     if msg.vel_left >= 0:
    #         self.left_wheel.direction = FORWARD
    #     else:
    #         self.left_wheel.direction = BACKWARD
    #     if msg.vel_right >= 0:
    #         self.right_wheel.direction = FORWARD
    #     else:
    #         self.right_wheel.direction = BACKWARD
    
    def cb_camera_info(self, msg):

        # self.logdebug("camera info received! ")
        if not self.camera_info_received:
            self.camera_info = msg
            self.rectifier = Rectify(msg)
            self.camera_P = np.array(msg.P).reshape((3,4))
            self.camera_K = np.array(msg.K).reshape((3,3))
            self.at_camera_params = (self.camera_P[0,0], self.camera_P[1,1],
                                     self.camera_P[0,2], self.camera_P[1,2])
            self.tf_cameraFbaselink = homography2transformation(self.homography_g2p, self.camera_K)
            self.log(f"tf_cameraFbaselink is {self.tf_cameraFbaselink}")
            self.camera_info_received = True

        return

    def cb_compressed_image(self, msg):
        # only localize once if apritag detected 
        if not self.camera_info_received:
            self.log("Image received before camera info received. Waiting for camera info...")
            return 

        # 1. process and rectify image
        cv2_img = self.bridge.compressed_imgmsg_to_cv2(msg)
        rect_img = self.rectifier.rectify(cv2_img)
        
        # detect tags
        self.detect(rect_img)
        
        return 
    # TODO: implement continuous pub function
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.first_loc:
                # DEBUG
            
                # self.broadcast_tf(self.tf_mapFcamera,
                #                 rospy.Time.now(),
                #                 "camera",
                #                 "map")

                # self.broadcast_tf(self.tf_mapFapriltag,
                #                 rospy.Time.now(),
                #                 "apriltag",
                #                 "map"
                # )

                # self.broadcast_tf(self.tf_mapFbaselink, 
                #                 rospy.Time.now(),
                #                 "encoder_baselink",
                #                 "map")

                # self.broadcast_tf(np.linalg.inv(self.tf_cameraFbaselink), # camera to baselink
                #                 rospy.Time.now(),
                #                 "camera",
                #                 "encoder_baselink")
                self.odm.run_update_pose()
                pose_baselink_in_map = self.odm.get_baselink_matrix()
                self.broadcast_tf(
                    pose_baselink_in_map,
                    rospy.Time.now(),
                    "encoder_baselink",
                    "map"
                )
                
            rate.sleep()

if __name__ == '__main__':
    node = EncoderLocNode(node_name='encoder_localization')
    # Keep it spinning to keep the node alive

    
    node.run()

    rospy.spin()
    rospy.loginfo("wheel_encoder_node is up and running...")

