#include "visualization.h"
#include <pcl_conversions/pcl_conversions.h>

using namespace Eigen;

ros::Publisher pub_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;

void register_pub(ros::NodeHandle &n)
{
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud2>("point_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
}

void pubOdometry(Estimator::Ptr estimator, double time)
{
    if (estimator->frontend->status == FrontendStatus::TRACKING_GOOD)
    {
        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(time);
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        SE3 pose = estimator->frontend->current_frame->Pose().inverse();
        Vector3d T = pose.translation();
        Quaterniond R = pose.unit_quaternion();
        Vector3d velocity = estimator->frontend->current_frame->Velocity();
        odometry.pose.pose.position.x = T.x();
        odometry.pose.pose.position.y = T.y();
        odometry.pose.pose.position.z = T.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        odometry.twist.twist.linear.x = velocity.x();
        odometry.twist.twist.linear.y = velocity.y();
        odometry.twist.twist.linear.z = velocity.z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(time);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header.stamp = ros::Time(time);
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);
    }
}

void pubKeyPoses(Estimator::Ptr estimator, double time)
{
    if (estimator->map->GetActiveKeyFrames().size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header.stamp = ros::Time(time);
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (auto key_frame : estimator->map->GetActiveKeyFrames())
    {
        geometry_msgs::Point pose_marker;
        pose_marker.x = key_frame.second->Pose().translation().x();
        pose_marker.y = key_frame.second->Pose().translation().y();
        pose_marker.z = key_frame.second->Pose().translation().z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubPointCloud(Estimator::Ptr estimator, double time)
{
    sensor_msgs::PointCloud2 ros_cloud;
    PointCloudRGB pcl_cloud;
    for (auto &map_point : estimator->map->GetActiveMapPoints())
    {
        PointRGB p;
        Vector3d pos = map_point.second->Pos();
        p.x = pos.x();
        p.y = pos.y();
        p.z = pos.z();
        //NOTE: semantic map
        LabelType label = map_point.second->label;
        switch (label)
        {
        case LabelType::Car:
            p.rgba = 0xFF0000FF;
            break;
        case LabelType::Person:
            p.rgba = 0x0000FFFF;
            break;
        case LabelType::Truck:
            p.rgba = 0xFF0000FF;
            break;
        default:
            p.rgba = 0x00FF00FF;
        }
        pcl_cloud.push_back(p);
    }
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = ros::Time(time);
    ros_cloud.header.frame_id = "world";
    pub_point_cloud.publish(ros_cloud);
}