cmake_minimum_required(VERSION 2.8.3)
project(fused_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  duckietown_msgs
  message_generation
  sensor_msgs
  std_msgs
  geometry_msgs
)

add_service_files(
  FILES
  UpdatePose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  duckietown_msgs
  geometry_msgs
  sensor_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# install(PROGRAMS
#   scripts/rqt_navigation
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )