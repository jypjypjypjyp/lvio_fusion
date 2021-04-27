#include <GeographicLib/LocalCartesian.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <termios.h>

#include "lvio_fusion/adapt/agent.h"
#include "lvio_fusion/adapt/environment.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/estimator.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion_node/CreateEnv.h"
#include "lvio_fusion_node/Init.h"
#include "lvio_fusion_node/Step.h"
#include "lvio_fusion_node/UpdateWeights.h"
#include "parameters.h"
#include "visualization.h"

#include "lvio_fusion/ceres/base.hpp"

using namespace std;

Estimator::Ptr estimator;

ros::Subscriber sub_imu, sub_lidar, sub_navsat, sub_img0, sub_img1, sub_objects, sub_eskf;
ros::Publisher pub_detector;
ros::ServiceServer svr_create_env, svr_step;
ros::ServiceClient clt_init, clt_update_weights;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<geometry_msgs::PoseStamped> odom_buf;
GeographicLib::LocalCartesian geo_converter;
mutex m_img_buf, m_odom_buf;
double delta_time = 0;
double init_time = 0;

// requisite topic
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if (init_time == 0)
    {
        delta_time = ros::Time::now().toSec() - img_msg->header.stamp.toSec();
        init_time = img_msg->header.stamp.toSec();
    }
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
    cv::Mat image;
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
        image = ptr->image.clone();
    }
    else if (img_msg->encoding == "bgr8")
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(ptr->image, image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        image = ptr->image.clone();
    }

    cv::equalizeHist(image, image);
    return image;
}

SE3d get_pose_from_path(double timestamp)
{
    static bool has_first_pose = false;
    static SE3d first_pose;
    SE3d pose;
    while (!odom_buf.empty())
    {
        std::unique_lock<std::mutex> lock(m_odom_buf);
        auto pose_stamp = odom_buf.front();
        odom_buf.pop();
        if (odom_buf.empty() || pose_stamp.header.stamp.toSec() > timestamp)
        {
            auto q = Quaterniond(pose_stamp.pose.orientation.w,
                                 pose_stamp.pose.orientation.x,
                                 pose_stamp.pose.orientation.y,
                                 pose_stamp.pose.orientation.z);
            auto t = Vector3d(pose_stamp.pose.position.x,
                              pose_stamp.pose.position.y,
                              pose_stamp.pose.position.z);
            if (!has_first_pose)
            {
                first_pose = SE3d(q, t);
                has_first_pose = true;
            }
            pose = SE3d(q, t) * first_pose.inverse();
        }
    }
    return pose;
}

// extract images with same timestamp from two topics
void sync_process()
{
    int n = 0;
    while (ros::ok())
    {
        cv::Mat image0, image1;
        std_msgs::Header header;
        double time = 0;
        if (!img0_buf.empty() && !img1_buf.empty())
        {
            m_img_buf.lock();
            double time0 = img0_buf.front()->header.stamp.toSec();
            double time1 = img1_buf.front()->header.stamp.toSec();
            if (time0 < time1 - 5 * epsilon)
            {
                img0_buf.pop();
                printf("throw img0\n");
                m_img_buf.unlock();
            }
            else if (time0 > time1 + 5 * epsilon)
            {
                img1_buf.pop();
                printf("throw img1\n");
                m_img_buf.unlock();
            }
            else
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image0 = get_image_from_msg(img0_buf.front());
                image1 = get_image_from_msg(img1_buf.front());
                img0_buf.pop();
                img1_buf.pop();
                m_img_buf.unlock();
                estimator->InputImage(time, image0, image1, get_pose_from_path(time));
                publish_car_model(estimator, time);
            }
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
    estimator->InputImu(t, acc, gyr);
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

void eskf_callback(const geometry_msgs::PoseStampedConstPtr &fused_msg)
{
    m_odom_buf.lock();
    odom_buf.push(*fused_msg);
    m_odom_buf.unlock();
}

void tf_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_tf(estimator, timer_event.current_real.toSec() - delta_time);
}

void pc_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_point_cloud(estimator, timer_event.current_real.toSec() - delta_time);
}

void lm_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_local_map(estimator, timer_event.current_real.toSec() - delta_time);
}

void od_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_odometry(estimator, timer_event.current_real.toSec() - delta_time);
}

void navsat_timer_callback(const ros::TimerEvent &timer_event)
{
    publish_navsat(estimator, timer_event.current_real.toSec() - delta_time);
}

bool create_env_callback(lvio_fusion_node::CreateEnv::Request &req,
                         lvio_fusion_node::CreateEnv::Response &res)
{
    res.id = Environment::Create(res.obs);
    return true;
}

bool step_callback(lvio_fusion_node::Step::Request &req,
                   lvio_fusion_node::Step::Response &res)
{
    Observation obs;
    Weights weights;
    weights.visual = req.visual;
    weights.lidar_ground = req.lidar_ground;
    weights.lidar_surf = req.lidar_surf;
    Environment::Step(req.id, weights, res.obs, &res.reward, (bool *)&res.done);
    return true;
}

// For non-blocking keyboard inputs
int getch(void)
{
    int ch;
    struct termios oldt;
    struct termios newt;

    // Store old settings, and copy to new settings
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;

    // Make required changes and apply the settings
    newt.c_lflag &= ~(ICANON | ECHO);
    newt.c_iflag |= IGNBRK;
    newt.c_iflag &= ~(INLCR | ICRNL | IXON | IXOFF);
    newt.c_lflag &= ~(ICANON | ECHO | ECHOK | ECHOE | ECHONL | ISIG | IEXTEN);
    newt.c_cc[VMIN] = 1;
    newt.c_cc[VTIME] = 0;
    tcsetattr(fileno(stdin), TCSANOW, &newt);

    // Get the current character
    ch = getchar();

    // Reapply old settings
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

    return ch;
}

void write_result(Estimator::Ptr estimator)
{
    ROS_WARN("Writing result file: %s", result_path.c_str());
    ofstream out(result_path, ios::out);
    out.setf(ios::fixed, ios::floatfield);
    out.precision(5);
    for (auto &pair : lvio_fusion::Map::Instance().keyframes)
    {
        out << pair.first - init_time << ",";
        SE3d pose = pair.second->pose;
        Vector3d T = pose.translation();
        Quaterniond R = pose.unit_quaternion();
        out << T.x() << ","
            << T.y() << ","
            << T.z() << ","
            << R.x() << ","
            << R.y() << ","
            << R.z() << ","
            << R.w() << endl;
    }
    out.close();
    ROS_WARN("Finished!!!");
}

void read_ground_truth()
{
    ROS_WARN("Reading ground truth file: %s", ground_truth_path.c_str());
    ifstream in(ground_truth_path);
    string line;
    stringstream ss;
    double time, x, y, z, qx, qy, qz, qw;
    double dt = lvio_fusion::Map::Instance().keyframes.begin()->first;
    Matrix3d R_tf;
    R_tf << 0, 0, 1,
        -1, 0, 0,
        0, -1, 0;
    Quaterniond q_tf(R_tf);
    auto RRR = ypr2R(Vector3d(90, -90, 0));
    Quaterniond qqq(RRR);
    SE3d tf(q_tf, Vector3d::Zero()); // tum ground truth to lvio_fusion
    if (in)
    {
        while (getline(in, line))
        {
            ss << line;
            ss >> time >> x >> y >> z >> qx >> qy >> qz >> qw;
            cout << time << "," << x << "," << y << "," << z << "," << qx << "," << qy << "," << qz << "," << qw << endl;
            auto a = SE3d(Quaterniond(qw, qx, qy, qz), Vector3d(x, y, z));
            a.so3() = a.so3() * SO3d(q_tf.inverse());
            Environment::ground_truths[dt + time] = tf * a;
            ss.clear();
        }
    }
    else
    {
        ROS_ERROR("No such file");
    }
}

void start_train()
{
    ROS_WARN("Start Training!");
    Environment::Init(estimator);
    // call init server
    clt_init.waitForExistence();
    ROS_WARN("Initialization Finished");
    lvio_fusion_node::Init srv;
    if (!clt_init.call(srv) || !srv.response.status)
    {
        ROS_ERROR("Error: can not initialize rl env.");
    }
}

void keyboard_process()
{
    char key;
    while (ros::ok())
    {
        key = getch();
        if (estimator->frontend->status != FrontendStatus::TRACKING_GOOD)
            continue;
        switch (key)
        {
        case 's':
            write_result(estimator);
            ros::shutdown();
            break;
        case 't':
            if (train)
            {
                read_ground_truth();
                start_train();
            }
            break;
        case 'e':
        {
            double end_time = (--lvio_fusion::Map::Instance().keyframes.end())->first;
            lvio_fusion::Map::Instance().end = true;
            estimator->backend->UpdateMap();
        }
            ROS_WARN("Final Navsat Optimization!");
            break;
        default:
            break;
        }
    }
}

class RealCore : public lvio_fusion::Core
{
public:
    virtual void Update(Observation &obs, Weights &weights)
    {
        lvio_fusion_node::UpdateWeights srv;
        srv.request.obs = obs;
        if (!clt_update_weights.call(srv))
        {
            ROS_ERROR("Error: can not update weights.");
            return;
        }
        weights.visual = srv.response.visual;
        weights.lidar_ground = srv.response.lidar_ground;
        weights.lidar_surf = srv.response.lidar_surf;
        weights.updated = true;
    }
};

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
            printf("Please intput: rosrun lvio_fusion_node lvio_fusion_node [config file] \n"
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
            ROS_ERROR("Error: %s", config_file.c_str());
            return 1;
        }
        ROS_INFO("Load config_file: %s", config_file.c_str());
    }
    read_parameters(config_file);
    estimator = Estimator::Ptr(new Estimator(config_file));
    assert(estimator->Init(use_imu, use_lidar, use_navsat, use_loop, use_adapt) == true);
    ROS_WARN("Waiting for images...");
    register_pub(n);
    ros::Timer tf_timer = n.createTimer(ros::Duration(0.0001), tf_timer_callback);
    ros::Timer od_timer = n.createTimer(ros::Duration(1), od_timer_callback);
    ros::Timer lm_timer = n.createTimer(ros::Duration(0.1), lm_timer_callback);
    ros::Timer pc_timer;
    ros::Timer navsat_timer;

    cout << "image0:" << IMAGE0_TOPIC << endl;
    sub_img0 = n.subscribe(IMAGE0_TOPIC, 10, img0_callback);
    cout << "image1:" << IMAGE1_TOPIC << endl;
    sub_img1 = n.subscribe(IMAGE1_TOPIC, 10, img1_callback);
    if (use_imu)
    {
        cout << "imu:" << IMU_TOPIC << endl;
        sub_imu = n.subscribe(IMU_TOPIC, 100, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    if (use_lidar)
    {
        cout << "lidar:" << LIDAR_TOPIC << endl;
        sub_lidar = n.subscribe(LIDAR_TOPIC, 10, lidar_callback);
        pc_timer = n.createTimer(ros::Duration(10), pc_timer_callback);
    }
    if (use_navsat)
    {
        cout << "navsat:" << NAVSAT_TOPIC << endl;
        sub_navsat = n.subscribe(NAVSAT_TOPIC, 10, navsat_callback);
        navsat_timer = n.createTimer(ros::Duration(1), navsat_timer_callback);
    }
    if (use_eskf)
    {
        sub_eskf = n.subscribe("/eskf_fusion_node/fused_odom", 10, eskf_callback);
    }
    if (use_adapt)
    {
        clt_update_weights = n.serviceClient<lvio_fusion_node::UpdateWeights>("/lvio_fusion_node/update_weight");
        Agent::SetCore(new RealCore());
    }
    if (train)
    {
        clt_init = n.serviceClient<lvio_fusion_node::Init>("/lvio_fusion_node/init");
        svr_create_env = n.advertiseService("/lvio_fusion_node/create_env", create_env_callback);
        svr_step = n.advertiseService("/lvio_fusion_node/step", step_callback);
    }
    thread sync_thread{sync_process};
    thread control_thread{keyboard_process};
    ros::spin();
    sync_thread.join();
    control_thread.join();
    return 0;
}
