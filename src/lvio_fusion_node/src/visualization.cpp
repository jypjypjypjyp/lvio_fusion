#include "visualization.h"
#include <pcl_conversions/pcl_conversions.h>

using namespace Eigen;

ros::Publisher path_pub;
ros::Publisher navsat_pub;
ros::Publisher points_cloud_pub;
ros::Publisher points_cloud_pub1;
nav_msgs::Path path, navsat_path;

void register_pub(ros::NodeHandle &n)
{
    path_pub = n.advertise<nav_msgs::Path>("path", 1000);
    navsat_pub = n.advertise<nav_msgs::Path>("navsat_path", 1000);
    points_cloud_pub = n.advertise<sensor_msgs::PointCloud2>("point_cloud", 1000);
    points_cloud_pub1 = n.advertise<sensor_msgs::PointCloud2>("point_cloud1", 1000);
}

void pub_odometry(Estimator::Ptr estimator, double time)
{
    if (estimator->frontend->status == FrontendStatus::TRACKING_GOOD)
    {
        path.poses.clear();
        for (auto frame : estimator->map->GetAllKeyFrames())
        {
            auto position = frame.second->pose.inverse().translation();
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time(frame.first);
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose.position.x = position.x();
            pose_stamped.pose.position.y = position.y();
            pose_stamped.pose.position.z = position.z();
            path.poses.push_back(pose_stamped);
        }
        path.header.stamp = ros::Time(time);
        path.header.frame_id = "world";
        path_pub.publish(path);
    }
    sensor_msgs::PointCloud2 ros_cloud;
    PointCloudRGB pcl_cloud;
    for (auto frame : estimator->map->GetAllKeyFrames())
    {
        auto position = frame.second->pose.inverse().translation();
        PointRGB p;
        p.x = position.x();
        p.y = position.y();
        p.z = position.z();
        p.rgba = 0x00FF00FF;
        pcl_cloud.push_back(p);
    }
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = ros::Time(time);
    ros_cloud.header.frame_id = "world";
    points_cloud_pub1.publish(ros_cloud);
}

void pub_navsat(Estimator::Ptr estimator, double time)
{
    if (estimator->map->navsat_map->initialized)
    {
        if (navsat_path.poses.size() == 0)
        {
            for (auto mp_pair : estimator->map->navsat_map->navsat_points)
            {
                NavsatPoint point = mp_pair.second;
                geometry_msgs::PoseStamped pose_stamped;
                pose_stamped.header.stamp = ros::Time(point.time);
                pose_stamped.header.frame_id = "navsat";
                pose_stamped.pose.position.x = point.position.x();
                pose_stamped.pose.position.y = point.position.y();
                pose_stamped.pose.position.z = point.position.z();
                navsat_path.poses.push_back(pose_stamped);
            }
        }
        else
        {
            geometry_msgs::PoseStamped pose_stamped;
            NavsatPoint point = (--estimator->map->navsat_map->navsat_points.end())->second;
            pose_stamped.pose.position.x = point.position.x();
            pose_stamped.pose.position.y = point.position.y();
            pose_stamped.pose.position.z = point.position.z();
            navsat_path.poses.push_back(pose_stamped);
        }
        navsat_path.header.stamp = ros::Time(time);
        navsat_path.header.frame_id = "navsat";
        navsat_pub.publish(navsat_path);
    }
}

void pub_tf(Estimator::Ptr estimator, double time)
{
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion tf_q;
    tf::Vector3 tf_t;
    // base_link
    if (estimator->frontend->status == FrontendStatus::TRACKING_GOOD)
    {
        SE3d pose = estimator->frontend->current_frame->pose;
        Quaterniond pose_q = pose.unit_quaternion();
        Vector3d pose_t = pose.translation();
        tf_q.setValue(pose_q.w(), pose_q.x(), pose_q.y(), pose_q.z());
        tf_t.setValue(pose_t.x(), pose_t.y(), pose_t.z());
        transform.setOrigin(tf_t);
        transform.setRotation(tf_q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time(time), "world", "base_link"));
    }
    // navsat
    if (estimator->map->navsat_map != nullptr && estimator->map->navsat_map->initialized)
    {
        double *tf_data = estimator->map->navsat_map->tf.data();
        tf_q.setValue(tf_data[0], tf_data[1], tf_data[2], tf_data[3]);
        tf_t.setValue(tf_data[4], tf_data[5], tf_data[6]);
        transform.setOrigin(tf_t);
        transform.setRotation(tf_q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time(time), "world", "navsat"));
    }
}

void pub_point_cloud(Estimator::Ptr estimator, double time)
{
    sensor_msgs::PointCloud2 ros_cloud;
    PointCloudRGB pcl_cloud;
    static std::unordered_map<unsigned long, Vector3d> position_cache;
    for (auto kf_pair : estimator->map->GetActiveKeyFrames(estimator->backend->ActiveTime()))
    {
        auto frame = kf_pair.second;
        auto features = frame->features_right;
        for (auto feature_pair : features)
        {
            if (!feature_pair.second->mappoint.expired())
            {
                auto landmark = feature_pair.second->mappoint.lock();
                position_cache[landmark->id] = landmark->ToWorld();
            }
        }
    }

    auto landmarks = estimator->map->GetAllMapPoints();
    for (auto point_pair_iter = position_cache.begin(); point_pair_iter != position_cache.end();)
    {
        auto landmark_iter = landmarks.find(point_pair_iter->first);
        if (landmark_iter == landmarks.end())
        {
            point_pair_iter = position_cache.erase(point_pair_iter);
            continue;
        }

        PointRGB p;
        Vector3d pos = point_pair_iter->second;
        p.x = pos.x();
        p.y = pos.y();
        p.z = pos.z();
        //NOTE: semantic map
        LabelType label = landmark_iter->second->label;
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
        point_pair_iter++;
    }
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = ros::Time(time);
    ros_cloud.header.frame_id = "world";
    points_cloud_pub.publish(ros_cloud);
}
