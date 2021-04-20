#ifndef lvio_fusion_VISUALIZATION_H
#define lvio_fusion_VISUALIZATION_H

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>

#include "lvio_fusion/estimator.h"

using namespace lvio_fusion;

void register_pub(ros::NodeHandle &n);

void publish_odometry(Estimator::Ptr estimator, double time);

void publish_navsat(Estimator::Ptr estimator, double time);

void publish_point_cloud(Estimator::Ptr estimator, double time);

void publish_tf(Estimator::Ptr estimator, double time);

void publish_car_model(Estimator::Ptr estimator, double time);

#endif // lvio_fusion_VISUALIZATION_H