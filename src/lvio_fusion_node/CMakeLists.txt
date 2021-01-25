cmake_minimum_required(VERSION 3.10)
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
    object_detector
    image_transport
    pcl_conversions
    pcl_ros)
include_directories(${catkin_INCLUDE_DIRS})

# GeographicLib
find_package(GeographicLib REQUIRED)
include_directories(${GeographicLib_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
    ${catkin_LIBRARIES}
    ${GeographicLib_LIBRARIES}
    )

catkin_package()

add_subdirectory(src)
