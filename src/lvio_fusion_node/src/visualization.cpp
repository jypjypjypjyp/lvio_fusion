#include "visualization.h"
#include "camera_pose.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.h"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.h>   //CompressedImage
#include <image_transport/image_transport.h>   //CompressedImage
#include <cv_bridge/cv_bridge.h>              //CompressedImage

ros::Publisher pub_path;
ros::Publisher pub_navsat;
ros::Publisher pub_points_cloud;
ros::Publisher pub_local_map;
ros::Publisher pub_car_model;
ros::Publisher pub_car_model_navigation;//NAVI
ros::Publisher pub_plan_path;//NAVI
ros::Publisher pub_navigation;//NAVI
ros::Publisher pub_CompressedImage0,pub_CompressedImage1;//CompressedImage
nav_msgs::Path path, navsat_path, plan_path;//NAVI
ros::Publisher pub_camera_pose_visual;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);

void register_pub(ros::NodeHandle &n)
{
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_navsat = n.advertise<nav_msgs::Path>("navsat_path", 1000);
    pub_points_cloud = n.advertise<sensor_msgs::PointCloud2>("point_cloud", 1000);
    pub_local_map = n.advertise<sensor_msgs::PointCloud2>("local_map", 1000);
    pub_car_model_navigation = n.advertise<visualization_msgs::Marker>("car_model_navigation", 1000);//NAVI
    pub_navigation = n.advertise<nav_msgs::OccupancyGrid>("grid_map", 1);//NAVI
    pub_plan_path = n.advertise<nav_msgs::Path>("plan_path", 1000);//NAVI
    pub_car_model = n.advertise<visualization_msgs::Marker>("car_model", 1000);
    pub_CompressedImage0=n.advertise<sensor_msgs::CompressedImage>("CompressedImage0",1000);//CompressedImage
    pub_CompressedImage1=n.advertise<sensor_msgs::CompressedImage>("CompressedImage1",1000);//CompressedImage

    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

void publish_odometry(Estimator::Ptr estimator, double time)
{
    auto &&submap = PoseGraph::Instance().GetSections(0, 0);
    submap[PoseGraph::Instance().current_section.A] = PoseGraph::Instance().current_section;
    path.poses.clear();
    cameraposevisual.reset();
    for (auto &pair : lvio_fusion::Map::Instance().keyframes)
    {
        auto pose = pair.second->pose;
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(pair.first);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = pose.translation().x();
        pose_stamped.pose.position.y = pose.translation().y();
        pose_stamped.pose.position.z = pose.translation().z();
        pose_stamped.pose.orientation.w = pose.unit_quaternion().w();
        pose_stamped.pose.orientation.x = pose.unit_quaternion().x();
        pose_stamped.pose.orientation.y = pose.unit_quaternion().y();
        pose_stamped.pose.orientation.z = pose.unit_quaternion().z();
        path.poses.push_back(pose_stamped);
        if (pair.first == submap.begin()->first)
        {
            geometry_msgs::PoseStamped pose_stamped_sec;
            pose_stamped_sec.header.stamp = ros::Time(pair.first);
            pose_stamped_sec.header.frame_id = "world";
            pose_stamped_sec.pose.position.x = pose.translation().x();
            pose_stamped_sec.pose.position.y = pose.translation().y();
            pose_stamped_sec.pose.position.z = pose.translation().z() + 10;
            path.poses.push_back(pose_stamped_sec);
            path.poses.push_back(pose_stamped);
            submap.erase(submap.begin());
        }
        if (pair.second->loop_closure)
        {
            auto position = pair.second->loop_closure->frame_old->GetPosition();
            geometry_msgs::PoseStamped pose_stamped_loop;
            pose_stamped_loop.header.stamp = ros::Time(pair.first);
            pose_stamped_loop.header.frame_id = "world";
            pose_stamped_loop.pose.position.x = position.x();
            pose_stamped_loop.pose.position.y = position.y();
            pose_stamped_loop.pose.position.z = position.z();
            path.poses.push_back(pose_stamped_loop);
            path.poses.push_back(pose_stamped);
        }
        SE3d left_camera_pose = pose * Camera::Get(0)->extrinsic;
        SE3d right_camera_pose = pose * Camera::Get(1)->extrinsic;
        cameraposevisual.add_pose(left_camera_pose.translation(), left_camera_pose.unit_quaternion());
        cameraposevisual.add_pose(right_camera_pose.translation(), right_camera_pose.unit_quaternion());
    }
    path.header.stamp = ros::Time(time);
    path.header.frame_id = "world";
    pub_path.publish(path);
    cameraposevisual.publish_by(pub_camera_pose_visual, path.header);
}

void publish_navsat(Estimator::Ptr estimator, double time)
{
    auto navsat = Navsat::Get();
    static double finished = 0;
    static int i = 0;
    if (navsat->initialized)
    {
        auto iter = navsat->raw.upper_bound(finished);
        while (++iter != navsat->raw.end())
        {
            if (++i % 10 == 0)
            {
                geometry_msgs::PoseStamped pose_stamped;
                Vector3d point = navsat->GetPoint(iter->first);
                pose_stamped.header.stamp = ros::Time(iter->first);
                pose_stamped.header.frame_id = "world";
                pose_stamped.pose.position.x = point.x();
                pose_stamped.pose.position.y = point.y();
                pose_stamped.pose.position.z = point.z();
                navsat_path.poses.push_back(pose_stamped);
            }
        }
        finished = (--iter)->first;
        navsat_path.header.stamp = ros::Time(time);
        navsat_path.header.frame_id = "world";
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
    if (estimator->frontend->status == FrontendStatus::TRACKING)
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
    if (Navsat::Num() && Navsat::Get()->initialized)
    {
        double *tf_data = Navsat::Get()->extrinsic.data();
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

void publish_local_map(Estimator::Ptr estimator, double time)
{
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(estimator->frontend->local_map.GetLocalLandmarks(), ros_cloud);
    ros_cloud.header.stamp = ros::Time(time);
    ros_cloud.header.frame_id = "world";
    pub_local_map.publish(ros_cloud);
}

void publish_car_model(Estimator::Ptr estimator, double time)
{
    visualization_msgs::Marker car_mesh;
    car_mesh.header.stamp = ros::Time(time);
    car_mesh.header.frame_id = "world";
    car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::Marker::ADD;
    // car_mesh.mesh_resource = "package://lvio_fusion_node/models/car.dae";
    car_mesh.mesh_resource = "file:///home/zoet/Projects/lvio_fusion/src/lvio_fusion_node/models/car.dae";
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

    float major_scale = 1.0;
    car_mesh.scale.x = major_scale;
    car_mesh.scale.y = major_scale;
    car_mesh.scale.z = major_scale;

    pub_car_model.publish(car_mesh);
}

//NAVI
void publish_car_model_navigation(Estimator::Ptr estimator, double time)
{
    visualization_msgs::Marker car_mesh;
    car_mesh.header.stamp = ros::Time(time);
    car_mesh.header.frame_id = "navigation";
    car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::Marker::ADD;
    // car_mesh.mesh_resource = "package://lvio_fusion_node/models/car.dae";
    car_mesh.mesh_resource = "file:///home/zoet/Projects/lvio_fusion/src/lvio_fusion_node/models/car.dae";
    car_mesh.id = 0;

    SE3d pose = estimator->frontend->current_frame->pose;
    Matrix3d rotate;
    rotate << -1, 0, 0, 0, 0, 1, 0, 1, 0;
    Quaterniond Q;
    Q = pose.unit_quaternion() * rotate;
    Vector3d t =  pose.translation();
    car_mesh.pose.position.x = t.x();
    car_mesh.pose.position.y = t.y();
    car_mesh.pose.position.z = 0;
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

    pub_car_model_navigation.publish(car_mesh);
}

void publish_navigation(Estimator::Ptr estimator, double time)
{
    nav_msgs::OccupancyGrid grid_map_msg;
    grid_map_msg.header.frame_id="navigation";
    grid_map_msg.header.stamp =  ros::Time(time); 
    int h=estimator->gridmap->height;
    int w=estimator->gridmap->width;
    grid_map_msg.info.resolution = estimator->gridmap->resolution;
    grid_map_msg.info.width = h;
    grid_map_msg.info.height = w;
    grid_map_msg.info.origin.position.x = (-h/2-0.5)* estimator->gridmap->resolution;
    grid_map_msg.info.origin.position.y = (-w/2-0.5)* estimator->gridmap->resolution;
    grid_map_msg.info.origin.position.z = 0;
    int p[w*h];
    cv::Mat grid_map = estimator ->gridmap->GetGridmap();
    std::vector<signed char> grid_map_vec;
    for(int row = 0; row < h; ++row)
    {
        for(int col = 0; col < w; ++col)
        {
            grid_map_vec.push_back(grid_map.at<char>(row, col));
        }
    }
    grid_map_msg.data=grid_map_vec;
    pub_navigation.publish(grid_map_msg);
}

void publish_plan_path(Estimator::Ptr estimator, double time)
{
    if(estimator->globalplanner->pathupdated)
    {
        plan_path.poses.clear();
        list<Vector2d> plan_path_ = estimator->globalplanner->GetPath();
 
        for( auto point: plan_path_)
        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time(time);
            pose_stamped.header.frame_id = "navigation";
            pose_stamped.pose.position.x = point.x();
            pose_stamped.pose.position.y = point.y();
            pose_stamped.pose.position.z = 0;
            plan_path.poses.push_back(pose_stamped);
        }
        plan_path.header.stamp = ros::Time(time);
        plan_path.header.frame_id = "navigation";
        LOG(INFO)<<"plan_path_: "<<plan_path_.size();
        pub_plan_path.publish(plan_path);
    }
}
//CompressedImage
sensor_msgs::CompressedImage cv_to_ros(cv::Mat image) // 将cv图像格式转换为ROS压缩图像格式
{
    std::string encoding = "bgr8";
    int bitDepth = sensor_msgs::image_encodings::bitDepth(encoding);
    int numChannels =sensor_msgs::image_encodings::numChannels(encoding);

    sensor_msgs::CompressedImage compressed;
    compressed.header = std_msgs::Header();
    compressed.format = "bgr8";

    std::vector<int> params;
    params.resize(9, 0);
    params[0] = 1;
    params[1] = 30;
    params[2] = 2;
    params[3] = 0;
    params[4] = 3;
    params[5] = 0;
    params[6] = 4;
    params[7] = 0;
    compressed.format += "; jpeg compressed ";

    if ((bitDepth == 8) || (bitDepth == 16))
    {
        // Target image format
        std::string targetFormat;
        if (sensor_msgs::image_encodings::isColor(encoding))
        {
            // convert color images to BGR8 format
            targetFormat = "bgr8";
            compressed.format += targetFormat;
        }
        cv::imencode(".jpg", image, compressed.data, params);
    }

    return compressed;
}
void publish_CompressedImage0(cv::Mat image)
{
        pub_CompressedImage0.publish(cv_to_ros(image));
}
void publish_CompressedImage1(cv::Mat image)
{
        pub_CompressedImage1.publish(cv_to_ros(image));
}