cmake_minimum_required(VERSION 3.10)
project(lvio_fusion_node)

set(CMAKE_BUILD_TYPE Debug)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    tf
    cv_bridge
    image_transport
    pcl_conversions
    pcl_ros
    lvio_fusion
    message_generation)
include_directories(${catkin_INCLUDE_DIRS})

# server
add_service_files(
    FILES 
        CreateEnv.srv
        Step.srv
        Init.srv
        UpdateWeights.srv
    )

generate_messages(
    DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
    )

# GeographicLib
find_package(GeographicLib REQUIRED)
include_directories(${GeographicLib_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
    ${catkin_LIBRARIES}
    ${GeographicLib_LIBRARIES}
    )

catkin_package(
    LIBRARIES lvio_fusion_node
    )

add_subdirectory(src)
