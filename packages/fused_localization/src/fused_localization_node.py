#!/usr/bin/env python3

import os
import yaml
import rospy
import sys
import numpy as np
from rospy.core import logerr
import tf
import cv2 
import enum
import time

from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from cv_bridge import CvBridge, CvBridgeError
from rospy import Subscriber, Publisher
from tf import TransformBroadcaster, TransformListener
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from dt_apriltags import Detector

from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion, Point
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped
from fused_localization.srv import UpdatePose, UpdatePoseResponse
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import CompressedImage, CameraInfo

# local import 
from utils.rectification import Rectify
from utils.wheel_odometry import WheelOdometry

fuse_state = enum.Enum("FuseState", ["USE_WHEEL", "USE_AT"])



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


class FusedLocNode(DTROS):

    def __init__(self, node_name):

################################ params, variables and flags ####################################
        # Initialize the DTROS parent class
        super(FusedLocNode, self).__init__(node_name, NodeType.GENERIC)
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

        self.fuse_state = fuse_state.USE_WHEEL # use wheel from start 
        self.last_at_detected = time.time()

        ####################### transforms declaration ##########################
        # 1. pre_defined tfs 
        # from apriltag to map, simple translation
        self.tf_mapFapriltag = np.array([   
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.09], # 9cm
            [0.0, 0.0, 0.0, 1.0]
        ]) 
        # to rotate the pose of camera & apriltags to meet the outputrequirement
        self.ori_cameraFoutput_camera = np.array([   # rotate camera to meet the output requirement
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]
        ]) 
        self.tf_output_apriltagFori_apriltag = np.array([   # rotate apriltag to meet the output requirement
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.tf_encoder_baselinkFat_baselink = np.array([   # assume encoder baselink and at baselink overlap at the beginning
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]
        ])

        # self.initial_mapFbaselink = None # should be defined in encoder_localization node 

        # 2. tfs available after receiving camera_info 
        self.tf_mapFencoder_baselink = None # from encoder baselink to map, provided by encoder_localization node 
        # self.tf_mapFcamera = None # from camera to map 
        self.tf_cameraFbaselink = None # from at_baselink to camera
        self.tf_baselinkFcamera = None # inverse of cameraFbaselink
        self.tf_mapFfused_baselink = None 
        self.tf_mapFoutput_camera = None 

        # 3. tfs available after first apriltag detection
        self.tf_cameraFapriltag = None 
        self.tf_apriltagFcamera = None # from camera to apriltag
        self.tf_mapFat_baselink = None 

        # 4. tfs available after apriltag missing 
        self.tf_at_baselinkFencoder_baselink = None 

############################# member objects needed to init before pub&sub ##################
        self.bridge = CvBridge()
        self.rectifier = None
        self.tf_bcaster = TransformBroadcaster()
        self.tf_listener = TransformListener()
        # apriltag detector
        self.at_detector = Detector(families='tag36h11',
                       nthreads=4,
                       quad_decimate=4.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

############################## subscribers and publishers ####################################
        
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

    def update_encoder_baselink(self):
        # get encoder pose from TF tree
        try:
            tvec, rvec = self.tf_listener.lookupTransform(
                "map",
                "encoder_baselink",
                rospy.Time()
            )
            pose_mat = tf.transformations.quaternion_matrix(rvec)
            pose_mat[:3, 3] = np.array(tvec)
            self.tf_mapFencoder_baselink = pose_mat

            if self.first_loc == False:
                self.first_loc = True

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e: 
            self.logwarn(e)
        return 

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

    # when apriltag reappears in camera, update encoder_baselink by call ros service 
    def call_srv_update_pose(self, pose_mat):
        self.logdebug(f"enter call_srv_update_pose, waiting for {f'/{self.veh_name}/encoder_localization_node/update_pose'}")
        
        # 1. compose pose_stamped message from pose_mat
        def _matrix_to_quaternion(r):
            T = np.array((
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 1)
            ), dtype=np.float64)
            T[0:3, 0:3] = r
            return tf.transformations.quaternion_from_matrix(T)
        
        rvec = _matrix_to_quaternion(pose_mat[:3,:3])
        tvec = pose_mat[:3, 3].reshape(-1).tolist()
        
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position = Point(*tvec)
        pose_stamped.pose.orientation = Quaternion(*rvec)
        # 2. call ros service with pose_stamped message as data
        rospy.wait_for_service(f'/{self.veh_name}/encoder_localization_node/update_pose')
        try:
            update_pose = rospy.ServiceProxy('add_two_ints', UpdatePose)
            ack = update_pose(pose_stamped)
            return ack
        except rospy.ServiceException as e:
            self.logwarn("Service call failed: %s"%e)

    def detect(self, img):

        greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(greyscale_img, True,
                            self.at_camera_params, self.at_tag_size)

        # only choose the first tag to compute tfs 
        count = 0
        # select the first tag
        for tag in tags: 
            tf_cameraFapriltag = np.concatenate(
                (
                    np.concatenate((tag.pose_R, tag.pose_t),axis=1),  # 3x4
                    np.array([[0.0,0.0,0.0,1.0]]) # 1x4
                ), 
                axis=0
            )

            self.tf_cameraFapriltag = tf_cameraFapriltag
            self.tf_apriltagFcamera = np.linalg.inv(self.tf_cameraFapriltag)
            # ATTENTION: should rotate the apriltag to meet the output requirement
            self.tf_apriltagFbaselink = self.tf_output_apriltagFori_apriltag @ self.tf_apriltagFcamera @ self.tf_cameraFbaselink
            self.tf_mapFat_baselink = self.tf_mapFapriltag @ self.tf_apriltagFbaselink

            count += 1

            break
                     
        return count

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
            self.tf_baselinkFcamera = np.linalg.inv(self.tf_cameraFbaselink)       
            # self.log(f"tf_cameraFbaselink is {self.tf_cameraFbaselink}")
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
        detect_count = self.detect(rect_img)
        
        # reset last_at_detected timer
        if detect_count > 0:  
            self.last_at_detected = time.time()

        self.logdebug(f"{detect_count} apriltags detected in image")
        
        # fuse state transfer
        if detect_count == 0:
            # apriltags disappear from camera
            # record wheel encoder
            if self.fuse_state == fuse_state.USE_AT:
                # 0. check the time elapse of no tag detected is long enough
                elapse_at_missing = time.time() - self.last_at_detected
                if elapse_at_missing > 0.3: 
                    # 1. record transform from at_baseline to encoder_baseline
                    self.tf_encoder_baselinkFat_baselink = np.linalg.inv(self.tf_mapFencoder_baselink) @ self.tf_mapFat_baselink

                    # 2. set fuse_state to USE_WHEEL
                    self.fuse_state = fuse_state.USE_WHEEL
            
                    self.log("Switch to using wheel encoder")

        if detect_count > 0:
            # apriltag reappear in camera, switch back to using at_baseline
            if self.fuse_state == fuse_state.USE_WHEEL:
                # 1. call ros service to update baselink pose in encoder node
                self.call_srv_update_pose(self.tf_mapFat_baselink)
                # 2. set fuse_state back to USE_AT
                self.fuse_state = fuse_state.USE_AT
                self.log("Switch to using apriltag detector")
        return 

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # update encoder baselink from encoder localization node 
            self.update_encoder_baselink()

            # initialization requires: 
            #     camera_info received to get tf_cameraFbaselink
            #     tf_mapFencoder_baselink from encoder localization node  
            if self.camera_info_received and self.first_loc:
                # using wheel encoder to localize
                if self.fuse_state == fuse_state.USE_WHEEL:
                    
                    self.tf_mapFfused_baselink = self.tf_mapFencoder_baselink @ self.tf_encoder_baselinkFat_baselink
                    self.broadcast_tf(
                        self.tf_mapFfused_baselink,
                        rospy.Time.now(),
                        "fused_baselink",
                        "map"
                    )

                elif self.fuse_state  == fuse_state.USE_AT:

                    self.broadcast_tf(
                        self.tf_mapFat_baselink,
                        rospy.Time.now(),
                        "at_baselink",
                        "map"
                    )

                    self.tf_mapFfused_baselink = self.tf_mapFat_baselink
                    self.broadcast_tf(
                        self.tf_mapFfused_baselink,
                        rospy.Time.now(),
                        "fused_baselink",
                        "map"
                    )
                else:
                    self.logerr(f"fuse state {self.fuse_state} not implemented!")
                

                # calculate and publish transform from camera to map
                self.tf_mapFoutput_camera = self.tf_mapFfused_baselink @ self.tf_baselinkFcamera @ self.ori_cameraFoutput_camera
                self.broadcast_tf(
                    np.linalg.inv(self.tf_mapFoutput_camera), # camera to baselink
                    rospy.Time.now(),
                    "camera",
                    "map")

            rate.sleep()

if __name__ == '__main__':
    node = FusedLocNode(node_name='at_localization')
    # Keep it spinning to keep the node alive

    
    node.run()

    rospy.spin()
    # rospy.loginfo("wheel_encoder_node is up and running...")

