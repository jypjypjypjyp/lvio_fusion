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
include_directories(${catkin_INCLUDE_DIRS})

# server
generate_messages(
    DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
    )
# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# Thread
find_package (Threads)

set(THIRD_PARTY_LIBS
    ${catkin_LIBRARIES}
    ${GeographicLib_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${CERES_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES navigation_node
)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
