#include <cv_bridge/cv_bridge.h>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>

#include "lvio_fusion/estimator.h"
#include "parameters.h"
#include "visualization.h"

using namespace std;

Estimator::Ptr estimator = nullptr;

ros::Subscriber sub_imu, sub_lidar, sub_img0, sub_img1;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
mutex m_buf;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
    while (1)
    {
        cv::Mat image0, image1;
        std_msgs::Header header;
        double time = 0;
        m_buf.lock();
        if (!img0_buf.empty() && !img1_buf.empty())
        {
            double time0 = img0_buf.front()->header.stamp.toSec();
            double time1 = img1_buf.front()->header.stamp.toSec();
            // if (time0 < time1)
            // {
            //     img0_buf.pop();
            //     printf("throw img0\n");
            // }
            // else if (time0 > time1)
            // {
            //     img1_buf.pop();
            //     printf("throw img1\n");
            // }
            // else
            // {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image0 = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
                image1 = getImageFromMsg(img1_buf.front());
                img1_buf.pop();
            // }
        }
        m_buf.unlock();
        if (!image0.empty())
        {
            estimator->InputImage(time, image0, image1);
            pubOdometry(estimator, time);
            // pubKeyPoses(estimator, time);
            // pubCameraPose(estimator, time);
            pubPointCloud(estimator, time);
        }

        chrono::milliseconds dura(2);
        this_thread::sleep_for(dura);
    }
}

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr &lidar_msg)
{
    double t = lidar_msg->header.stamp.toSec();
    PointCloud point_cloud;
    pcl::fromROSMsg(*lidar_msg, point_cloud);
    PointCloudPtr laser_cloud_in_ptr(new PointCloud(point_cloud));
    estimator->InputPointCloud(t, laser_cloud_in_ptr);
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator->InputIMU(t, acc, gyr);
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lvio_fusion_node");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    string config_file = "read_config_path_is_not_correct";

    if (argc > 1)
    {
        if (argc != 2)
        {
            printf("please intput: rosrun vins vins_node [config file] \n"
                   "for example: rosrun lvio_fusion_node lvio_fusion_node "
                   "~/Projects/lvio-fusion/src/lvio_fusion_node/config/kitti.yaml \n");
            return 1;
        }

        config_file = argv[1];
    }
    else
    {
        if (!n.getParam("config_file", config_file))
        {
            ROS_INFO("Error: %s\n", config_file.c_str());
            return 1;
        }
        ROS_INFO("load config_file: %s\n", config_file.c_str());
    }
    readParameters(config_file);
    estimator = Estimator::Ptr(new Estimator(config_file));
    assert(estimator->Init() == true);

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    if (use_imu)
    {
        cout << "imu:" << IMU_TOPIC << endl;
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    if (use_lidar)
    {
        cout << "lidar:" << LIDAR_TOPIC << endl;
        sub_lidar = n.subscribe(LIDAR_TOPIC, 100, lidar_callback);
    }
    cout << "image0:" << IMAGE0_TOPIC << endl;
    sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    if (num_of_cam == 2)
    {
        cout << "image1:" << IMAGE1_TOPIC << endl;
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    thread sync_thread{sync_process};
    ros::spin();
    return 0;
}
