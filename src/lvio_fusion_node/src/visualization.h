#ifndef lvio_fusion_VISUALIZATION_H
#define lvio_fusion_VISUALIZATION_H

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>

#include "CameraPoseVisualization.h"
#include "lvio_fusion/estimator.h"

using namespace lvio_fusion;

extern ros::Publisher pub_odometry;
extern ros::Publisher pub_path, pub_pose;
extern ros::Publisher pub_cloud, pub_map;
extern ros::Publisher pub_key_poses;
extern ros::Publisher pub_ref_pose, pub_cur_pose;
extern ros::Publisher pub_key;
extern nav_msgs::Path path;
extern ros::Publisher pub_pose_graph;
extern int IMAGE_ROW, IMAGE_COL;

void registerPub(ros::NodeHandle &n);

void pubOdometry(Estimator::Ptr estimator, double time);

void pubKeyPoses(Estimator::Ptr estimator, double time);

void pubCameraPose(Estimator::Ptr estimator, double time);

void pubPointCloud(Estimator::Ptr estimator, double time);

#endif // lvio_fusion_VISUALIZATION_H