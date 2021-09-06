cmake_minimum_required(VERSION 3.10)
project(navigation_node)

set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  tf
  cv_bridge
  image_transport
  message_generation
)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Thread
find_package (Threads)

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES navigation_node
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(

  ${catkin_INCLUDE_DIRS}
)
