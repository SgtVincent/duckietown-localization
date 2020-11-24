# code of kinematics adapted from duckietown tutorial Modelling of a differential drive vehicle
# on https://docs.duckietown.org/daffy/duckietown-robotics-development/out/representations_modeling.html 

from threading import Thread
import rospy 
import numpy as np
import time 

# updated wheel now have posisitve and negative wheels 
class WheelState:
    def __init__(self) -> None:
        self.ticks = 0
        self.msg_cnt = 0
        self.resolution = 135 # default 
        # self.direction = direction # deprecated 
        self.last_ticks = 0 # to compute delta 
        self.last_time = time.time() # to compute velocity 
        # self.distance = 0


class WheelOdometry:
    
    def __init__(
        self, 
        radius, 
        baseline, 
        initial_pose, # 4x4 numpy array which represents the pose of baselink in map frame
        logger=None,
        frequency=10):

        self.left_wheel = WheelState()
        self.right_wheel = WheelState()
        self.radius = radius
        self.baseline = baseline
        self.logger = logger
        self.r = rospy.Rate(frequency)
        self.last_pose_update = time.time()
        self.frequency = frequency

        # compute 2d representations from initial_pose
        self.cur_pos = initial_pose[:2, 3]
        # normalize the rotation matrix on x-y plane
        xy_pose_mat = initial_pose[:2, :2]
        U, S, Vh = np.linalg.svd(xy_pose_mat)
        xy_pose_mat = U @ Vh
        self.theta = np.arccos(xy_pose_mat[0,0])


    def update_wheel(self, wheel, msg):
        
        # self.logger.logdebug(f"WheelOdometry: update {wheel} with tick {msg.data}")
        
        if wheel == "left_wheel":
            self.left_wheel.ticks = msg.data
            self.left_wheel.resolution = msg.resolution 
            self.left_wheel.last_time = time.time() # not used yet 
            self.left_wheel.msg_cnt += 1
        elif wheel == "right_wheel":
            self.right_wheel.ticks = msg.data
            self.right_wheel.resolution = msg.resolution 
            self.right_wheel.last_time = time.time() # not used yet 
            self.right_wheel.msg_cnt += 1
        else:
            self.logger.logwarn(f"WheelOdometry: wheel name {wheel} not implemented")
        return 
        
    def get_next_pose(self, icc_pos, d, cur_theta, theta_displacement):
        """
        Compute the new next position in global frame
        Input:
            - icc_pos: numpy array of ICC position [x,y] in global frame
            - d: distance from robot to the center of curvature
            - cur_theta: current yaw angle in radian (float)
            - theta_displacement: the amount of angular displacement if we apply w for 1 time step
        Return:
            - next_position:
            - next_orientation:
        """
        
        # First, let's define the ICC frame as the frame centered at the location of ICC
        # and oriented such that its x-axis points towards the robot
        
        # Compute location of the point where the robot should be at (i.e., q)
        # in the frame of ICC.
        x_new_icc_frame = d * np.cos(theta_displacement)
        y_new_icc_frame = d * np.sin(theta_displacement)
        
        # Build transformation matrix from origin to ICC
        T_oc_angle = -(np.deg2rad(90) - cur_theta) # 
        icc_x, icc_y = icc_pos[0], icc_pos[1]
        T_oc = np.array([
            [np.cos(T_oc_angle), -np.sin(T_oc_angle), icc_x],
            [np.sin(T_oc_angle), np.cos(T_oc_angle), icc_y],
            [0, 0, 1]
        ]) # Transformation matrix from origin to the ICC
        
        # Build transformation matrix from ICC to the point where the robot should be at (i.e., q)
        T_cq = np.array([
            [1, 0, x_new_icc_frame],
            [0, 1, y_new_icc_frame],
            [0, 0, 1]
        ]) # Transformation matrix from ICC to the point where the robot should be at (i.e., q)
        
        # Convert the local point q to the global frame
        T_oq = np.dot(T_oc, T_cq) # Transformation matrix from origin to q
        
        next_position = np.array([T_oq[0,2], T_oq[1,2]])
        next_orientation = np.degrees(cur_theta) + np.degrees(theta_displacement)
        return next_position, next_orientation


    def drive(self, cur_pos, cur_angle, left_rate, right_rate, wheel_dist, wheel_radius, dt):
        """
        Input:
            - cur_pos: numpy array of current position [x,y] in global frame
            - cur_angle: current yaw angle in degree (float)
            - left_rate: turning rate of the left wheel in turns/sec(float)
            - right_rate: turning rate of the right wheel in turns/sec (float)
            - wheel_dist: distance between left and right wheels in meters (i.e., 2L) (float)
            - wheel_radius: radius of the wheels in meters (i.e., R) (float)
            - dt: time step (float)
        Return:
            - next_position: numpy array of next position [x,y] in global frame
            - next_orientation: next yaw angle ()
        """
        
        # Convert angle to radian and rename some variables
        cur_theta = np.deg2rad(cur_angle)
        l = wheel_dist
        
        # Convert turning rate (turns/sec) into (m/sec)
        # Note: the amount of distance traveled by 1 wheel revolution
        # is equal to its circumference (i.e., 2 * pi * radius)
        Vl = left_rate * 2. * np.pi * wheel_radius
        Vr = right_rate * 2. * np.pi * wheel_radius

        # If the wheel velocities are the same, then there is no rotation
        if Vl == Vr:
            v = Vl = Vr
            new_x = cur_pos[0] + dt * v * np.cos(cur_theta)
            new_y = cur_pos[1] + dt * v * np.sin(cur_theta)
            cur_pos = np.array([new_x, new_y])
            cur_angle = cur_angle # does not change since we are moving straight
            return cur_pos, cur_angle

        # Compute the angular rotation (i.e., theta_dot) velocity about the ICC (center of curvature)
        w = (Vr - Vl) / l
        
        # Compute the velocity (i.e., v_A)
        v = (Vr + Vl) / 2. 
        
        # Compute the distance from robot to the center of curvature (i.e., d)
        d = v / w 
        
        # Compute the amount of angular displacement if we apply w for 1 time step
        theta_displacement = w * dt 

        # Compute location of ICC in global frame
        icc_x = cur_pos[0] - d * (np.sin(cur_theta)) 
        icc_y = cur_pos[1] + d * (np.cos(cur_theta))
        icc_pos = np.array([icc_x, icc_y])
        
        # Compute next position and orientation given cx, cy, d, cur_theta, and theta_displacement
        next_position, next_orientation = self.get_next_pose(icc_pos, d, cur_theta, theta_displacement)
        
        return next_position, next_orientation
        

    def update_pose(self):
        # calculate delta of two wheels 
        left_delta_ticks = self.left_wheel.ticks - self.left_wheel.last_ticks
        self.left_wheel.last_ticks = self.left_wheel.ticks
        right_delta_ticks = self.right_wheel.ticks - self.right_wheel.last_ticks
        self.right_wheel.last_ticks = self.right_wheel.ticks
        
        # update time stamp
        cur_time = time.time()
        time_delta = cur_time - self.last_pose_update # in seconds
        self.last_pose_update = cur_time

        # calculate rates of two wheels and update position
        left_rate = (left_delta_ticks / self.left_wheel.resolution)/ time_delta 
        right_rate = (right_delta_ticks / self.right_wheel.resolution)/ time_delta 
        self.cur_pos, self.theta = self.drive(self.cur_pos, self.theta, left_rate, right_rate, 
            2*self.baseline, self.radius, time_delta)
        
        if self.logger:
            self.logger.logdebug(f"WheelOdometry: left/right rates are {left_rate},{right_rate}")
            self.logger.logdebug(f"WheelOdometry: cur_pos/angle are {self.cur_pos},{self.theta}")
    
    
    # convert the (x,y,theta) representation to 4x4 rotation matrix with all points in x-y plane 
    def get_baselink_matrix(self):
        tf_mat = np.array([
            [np.cos(self.theta), -np.sin(self.theta),   0,      self.cur_pos[0]],
            [np.sin(self.theta), np.cos(self.theta),    0,      self.cur_pos[1]],
            [0,                  0,                     1.0,    0              ],
            [0,                  0,                     0,      1.0            ] 
        ])
        return tf_mat 

    # wrapper for usage in main thread, will be deprecated once bug fixed
    def run_update_pose(self):
        if time.time() - self.last_pose_update > 1 / self.frequency:
            self.update_pose()

    # TODO: bugs to fix: main thread stuck in self.start() need to check python thread usage 

    # def start(self):
    #     self.thread = Thread(target=self.run())
    #     self.thread.start()
    #     return 

    # def run(self):
    #     # update pose estimation from ticks for every 1/frequency seconds
    #     while not rospy.is_shutdown():
    #         self.update_pose()
    #         self.r.sleep()