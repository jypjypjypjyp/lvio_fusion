#include "visualization.h"
#include <pcl_conversions/pcl_conversions.h>

using namespace Eigen;

ros::Publisher pub_path;
ros::Publisher pub_navsat;
ros::Publisher pub_points_cloud;
ros::Publisher pub_car_model;
nav_msgs::Path path, navsat_path;

void register_pub(ros::NodeHandle &n)
{
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_navsat = n.advertise<nav_msgs::Path>("navsat_path", 1000);
    pub_points_cloud = n.advertise<sensor_msgs::PointCloud2>("point_cloud", 1000);
    pub_car_model = n.advertise<visualization_msgs::Marker>("car_model", 1000);
}

void publish_odometry(Estimator::Ptr estimator, double time)
{
    if (estimator->frontend->status == FrontendStatus::TRACKING_GOOD)
    {
        path.poses.clear();
        for (auto frame : estimator->map->GetAllKeyFrames())
        {
            auto position = frame.second->pose.translation();
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time(frame.first);
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose.position.x = position.x();
            pose_stamped.pose.position.y = position.y();
            pose_stamped.pose.position.z = position.z();
            path.poses.push_back(pose_stamped);
            if (frame.second->loop_constraint)
            {
                auto position = frame.second->loop_constraint->frame_old->pose.translation();
                geometry_msgs::PoseStamped pose_stamped_loop;
                pose_stamped_loop.header.stamp = ros::Time(frame.first);
                pose_stamped_loop.header.frame_id = "world";
                pose_stamped_loop.pose.position.x = position.x();
                pose_stamped_loop.pose.position.y = position.y();
                pose_stamped_loop.pose.position.z = position.z();
                path.poses.push_back(pose_stamped_loop);
                path.poses.push_back(pose_stamped);
            }
        }
        path.header.stamp = ros::Time(time);
        path.header.frame_id = "world";
        pub_path.publish(path);
    }
}

void publish_navsat(Estimator::Ptr estimator, double time)
{
    if (estimator->map->navsat_map->initialized)
    {
        if (navsat_path.poses.size() == 0)
        {
            for (auto pair_mp : estimator->map->navsat_map->navsat_points)
            {
                NavsatPoint point = pair_mp.second;
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
        pub_navsat.publish(navsat_path);
    }
}

void publish_tf(Estimator::Ptr estimator, double time)
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

void publish_point_cloud(Estimator::Ptr estimator, double time)
{
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(estimator->mapping->GetGlobalMap(), ros_cloud);
    ros_cloud.header.stamp = ros::Time(time);
    ros_cloud.header.frame_id = "world";
    pub_points_cloud.publish(ros_cloud);
}

void publish_car_model(Estimator::Ptr estimator, double time)
{
    visualization_msgs::Marker car_mesh;
    car_mesh.header.stamp = ros::Time(time);
    car_mesh.header.frame_id = "world";
    car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::Marker::ADD;
    // car_mesh.mesh_resource = "package://lvio_fusion_node/models/car.dae";
    car_mesh.mesh_resource = "file:///home/jyp/Projects/lvio_fusion/src/lvio_fusion_node/models/car.dae";
    car_mesh.id = 0;

    SE3d pose = estimator->frontend->current_frame->pose;
    Matrix3d rotate;
    rotate << -1, 0, 0, 0, 0, 1, 0, 1, 0;
    Quaterniond Q;
    Q = pose.unit_quaternion() * rotate;
    Vector3d t = pose.translation();
    car_mesh.pose.position.x = t.x();
    car_mesh.pose.position.y = t.y();
    car_mesh.pose.position.z = t.z();
    car_mesh.pose.orientation.w = Q.w();
    car_mesh.pose.orientation.x = Q.x();
    car_mesh.pose.orientation.y = Q.y();
    car_mesh.pose.orientation.z = Q.z();

    car_mesh.color.a = 1.0;
    car_mesh.color.r = 1.0;
    car_mesh.color.g = 0.0;
    car_mesh.color.b = 0.0;

    float major_scale = 2.0;
    car_mesh.scale.x = major_scale;
    car_mesh.scale.y = major_scale;
    car_mesh.scale.z = major_scale;

    pub_car_model.publish(car_mesh);
}