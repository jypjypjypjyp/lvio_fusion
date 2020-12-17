cmake_minimum_required(VERSION 3.10)
project(rl_fusion)

find_package(catkin REQUIRED
  COMPONENTS
    roslint
    geometry_msgs
    sensor_msgs
    std_msgs
    message_generation)

# add_message_files(
#   FILES
#     BoundingBox.msg
#     BoundingBoxes.msg
# )

# generate_messages(
#   DEPENDENCIES
#     geometry_msgs
#     sensor_msgs
#     std_msgs
# )

catkin_package()

roslint_python()

include_directories()