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
from fused_localization.srv import UpdatePose, UpdatePoseResponse

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

        # initial pose: x=1.0, y=0, y=0, R=I
        self.tf_initial_mapFbaselink = np.array([
            [-1.0, 0.0,    0.0,   0.4],
            [0.0,  -1.0,   0.0,   0.0],
            [0.0,  0.0,    1.0,   0.0],
            [0.0,  0.0,    0.0,   1.0]
        ])
        self.tf_mapFcamera = None # from camera to map 
        self.tf_cameraFbaselink = None # from baselink to camera

        self.tf_mapFbaselink = None # target tf, from baselink to map 

############################# member objects needed to init before pub&sub ##################
        self.odm = WheelOdometry(self._radius, self._baseline, self.tf_initial_mapFbaselink, self, frequency=10)
        self.bridge = CvBridge()
        self.rectifier = None
        self.tf_bcaster = TransformBroadcaster()

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
        
        self.srv_update_pose =  rospy.Service(
            f'~/{self.node_name}/update_pose', 
            UpdatePose, 
            self.handle_update_pose)

        self.log(f"Start service {f'/{self.veh_name}/{self.node_name}/update_pose'}")
        

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
        # self.logdebug(f"Published encoder_baselink with rvec({rvec}), tvec({tvec}")
        self.tf_bcaster.sendTransform(
                                tvec.tolist(),
                                rvec.tolist(),
                                time,
                                child,
                                parent)
        return 

    def handle_update_pose(self, req):
        pose = req.pose_stamped.pose
        quat = pose.orientation 
        rvec = [quat.x, quat.y, quat.z, quat.w]
        point = pose.position
        tvec = [point.x, point.y, point.z]

        pose_mat = tf.transformations.quaternion_matrix(rvec)
        pose_mat[:3, 3] = np.array(tvec)

        self.odm.update_pose(pose_mat)
        return req.pose_stamped.header.seq # return sequence number as ACK 

    def cb_encoder_data_left(self, msg):
        self.odm.update_wheel("left_wheel", msg)
        return 
        

    def cb_encoder_data_right(self, msg):
        # self.logdebug(f"Main: cb_encoder_data_right called")
        self.odm.update_wheel("right_wheel", msg)
        pass

    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():


            # publish wheel odometry TF 
            self.odm.run_update_pose()
            pose_baselink_in_map = self.odm.get_baselink_matrix()
            self.broadcast_tf(
                pose_baselink_in_map,
                rospy.Time.now(),
                "encoder_baselink",
                "map"
            )
            rate.sleep()
   
            rate.sleep()

if __name__ == '__main__':
    node = EncoderLocNode(node_name='encoder_localization_node')
    # Keep it spinning to keep the node alive

    
    node.run()

    rospy.spin()
    # rospy.loginfo("wheel_encoder_node is up and running...")

