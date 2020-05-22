cmake_minimum_required(VERSION 2.8.3)
project(lvio_fusion_node)

set(CMAKE_BUILD_TYPE Debug)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    lvio_fusion
    image_transport
    pcl_conversions
    pcl_ros)
include_directories(${catkin_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Thread
find_package (Threads)

set(THIRD_PARTY_LIBS
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    )

catkin_package()

add_subdirectory(src)
