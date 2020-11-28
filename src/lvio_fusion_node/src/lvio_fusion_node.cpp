#include <GeographicLib/LocalCartesian.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>

#include "lvio_fusion/common.h"
#include "lvio_fusion/estimator.h"
#include "object_detector/BoundingBoxes.h"
#include "parameters.h"
#include "visualization.h"

using namespace std;

Estimator::Ptr estimator;

ros::Subscriber sub_imu, sub_lidar, sub_navsat, sub_img0, sub_img1, sub_objects;
ros::Publisher pub_detector;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
GeographicLib::LocalCartesian geo_converter;
object_detector::BoundingBoxesConstPtr obj_buf;
mutex m_img_buf, m_cond;
condition_variable cond;
double delta_time = 0;

// requisite topic
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    delta_time = ros::Time::now().toSec() - img_msg->header.stamp.toSec();
    m_img_buf.lock();
    img0_buf.push(img_msg);
    m_img_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_img_buf.lock();
    img1_buf.push(img_msg);
    m_img_buf.unlock();
}

cv::Mat get_image_from_msg(const sensor_msgs::ImageConstPtr &img_msg)
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

//NOTE： semantic map
void objects_callback(const object_detector::BoundingBoxesConstPtr &obj_msg)
{
    std::unique_lock<std::mutex> lk(m_cond);
    obj_buf = obj_msg;
    cond.notify_all();
}
const map<string, LabelType> map_label = {
    {"person", LabelType::Person},
    {"car", LabelType::Car},
    {"truck", LabelType::Truck},
};
vector<DetectedObject> get_objects_from_msg(const object_detector::BoundingBoxesConstPtr &obj_msg)
{
    vector<DetectedObject> objects;
    for (auto box : obj_msg->bounding_boxes)
    {
        if (map_label.find(box.Class) == map_label.end())
            continue;
        objects.push_back(DetectedObject(map_label.at(box.Class), box.probability, box.xmin, box.ymin, box.xmax, box.ymax));
    }
    return objects;
}

void write_result(Estimator::Ptr estimator, double time)
{
    ofstream foutC(result_path, ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << time * 1e9 << ",";
    foutC.precision(5);
    SE3d pose = estimator->frontend->current_frame->pose;
    Vector3d T = pose.translation();
    Quaterniond R = pose.unit_quaternion();
    foutC << T.x() << ","
          << T.y() << ","
          << T.z() << ","
          << R.w() << ","
          << R.x() << ","
          << R.y() << ","
          << R.z() << endl;
    foutC.close();
}

// extract images with same timestamp from two topics
void sync_process()
{
    int n = 0;
    while (1)
    {
        cv::Mat image0, image1;
        std_msgs::Header header;
        double time = 0;
        if (!img0_buf.empty() && !img1_buf.empty())
        {
            m_img_buf.lock();
            // double time0 = img0_buf.front()->header.stamp.toSec();
            // double time1 = img1_buf.front()->header.stamp.toSec();
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
            image0 = get_image_from_msg(img0_buf.front());
            image1 = get_image_from_msg(img1_buf.front());
            // }
            if (n++ % 7 == 0 && is_semantic)
            {
                pub_detector.publish(img0_buf.front());
                img0_buf.pop();
                img1_buf.pop();
                m_img_buf.unlock();

                std::unique_lock<std::mutex> lk(m_cond);
                cond.wait_for(lk, 200ms);
                if (obj_buf != nullptr)
                {
                    auto objects = get_objects_from_msg(obj_buf);
                    obj_buf;
                    // DEBUG
                    // for (auto object : objects)
                    // {
                    //     cv::rectangle(image0,
                    //                   cv::Rect2i(cv::Point2i(object.xmin, object.ymin), cv::Point2i(object.xmax, object.ymax)),
                    //                   cv::Scalar(0, 255, 0));
                    // }
                    // cv::imshow("debug", image0);
                    // cv::waitKey(2);
                    estimator->InputImage(time, image0, image1, objects);
                }
                else
                {
                    estimator->InputImage(time, image0, image1);
                }
            }
            else
            {
                img0_buf.pop();
                img1_buf.pop();
                m_img_buf.unlock();
                estimator->InputImage(time, image0, image1);
            }
            publish_car_model(estimator, time);
        }

        chrono::milliseconds dura(2);
        this_thread::sleep_for(dura);
    }
}

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr &lidar_msg)
{
    double t = lidar_msg->header.stamp.toSec();
    Point3Cloud point_cloud;
    pcl::fromROSMsg(*lidar_msg, point_cloud);
    Point3Cloud::Ptr laser_cloud_in_ptr(new Point3Cloud(point_cloud));
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

void navsat_callback(const sensor_msgs::NavSatFixConstPtr &navsat_msg)
{
    double t = navsat_msg->header.stamp.toSec();
    double latitude = navsat_msg->latitude;
    double longitude = navsat_msg->longitude;
    double altitude = navsat_msg->altitude;
    double pos_accuracy = navsat_msg->position_covariance[0];
    double xyz[3];
    static bool init = false;
    if (!init)
    {
        geo_converter.Reset(latitude, longitude, altitude);
        init = true;
    }
    geo_converter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);
    estimator->InputNavSat(t, xyz[0], xyz[1], xyz[2], pos_accuracy);
}

void tf_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_tf(estimator, timer_event.current_real.toSec() - delta_time);
}

void pc_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_point_cloud(estimator, timer_event.current_real.toSec() - delta_time);
}

void od_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_odometry(estimator, timer_event.current_real.toSec() - delta_time);
}

void navsat_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_navsat(estimator, timer_event.current_real.toSec() - delta_time);
}

int get_flags()
{
    int flags = 0;
    if (num_of_cam == 2)
        flags += Flag::Stereo;
    else if (num_of_cam == 1)
        flags += Flag::Mono;
    if (use_imu)
        flags += Flag::IMU;
    if (use_lidar)
        flags += Flag::Laser;
    if (use_navsat)
        flags += Flag::GNSS;
    if (is_semantic)
        flags += Flag::Semantic;
    return flags;
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
    read_parameters(config_file);
    estimator = Estimator::Ptr(new Estimator(config_file));
    assert(estimator->Init(use_imu, use_lidar, use_navsat, use_loop, is_semantic) == true);
    estimator->frontend->flags = get_flags();

    ROS_WARN("waiting for images...");

    register_pub(n);
    ros::Timer tf_timer = n.createTimer(ros::Duration(0.0001), tf_timer_callback);
    ros::Timer od_timer = n.createTimer(ros::Duration(1), od_timer_callback);
    ros::Timer pc_timer;
    ros::Timer navsat_timer;

    if (use_imu)
    {
        cout << "imu:" << IMU_TOPIC << endl;
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    if (use_lidar)
    {
        cout << "lidar:" << LIDAR_TOPIC << endl;
        sub_lidar = n.subscribe(LIDAR_TOPIC, 100, lidar_callback);
        pc_timer = n.createTimer(ros::Duration(10), pc_timer_callback);
    }
    if (use_navsat)
    {
        cout << "navsat:" << NAVSAT_TOPIC << endl;
        sub_navsat = n.subscribe(NAVSAT_TOPIC, 100, navsat_callback);
        navsat_timer = n.createTimer(ros::Duration(1), navsat_timer_callback);
    }
    cout << "image0:" << IMAGE0_TOPIC << endl;
    sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    if (num_of_cam == 2)
    {
        cout << "image1:" << IMAGE1_TOPIC << endl;
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    if (is_semantic)
    {
        sub_objects = n.subscribe("/object_detector/output_objects", 10, objects_callback);
        pub_detector = n.advertise<sensor_msgs::Image>("/object_detector/image_raw", 10);
    }
    thread sync_thread{sync_process};
    ros::spin();
    return 0;
}
