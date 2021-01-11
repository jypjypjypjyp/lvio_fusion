#include <GeographicLib/LocalCartesian.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl_conversions/pcl_conversions.h>
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
#include "lvio_fusion_node/BoundingBoxes.h"
#include "lvio_fusion_node/CreateEnv.h"
#include "lvio_fusion_node/Init.h"
#include "lvio_fusion_node/Step.h"
#include "parameters.h"
#include "visualization.h"

using namespace std;

Estimator::Ptr estimator;

ros::Subscriber sub_imu, sub_lidar, sub_navsat, sub_img0, sub_img1, sub_objects, sub_step;
ros::Publisher pub_detector;
ros::ServiceServer svr_create_env, svr_step;
ros::ServiceClient clt_init;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
GeographicLib::LocalCartesian geo_converter;
lvio_fusion_node::BoundingBoxesConstPtr obj_buf;
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
void objects_callback(const lvio_fusion_node::BoundingBoxesConstPtr &obj_msg)
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
vector<DetectedObject> get_objects_from_msg(const lvio_fusion_node::BoundingBoxesConstPtr &obj_msg)
{
    vector<DetectedObject> objects;
    for (auto &box : obj_msg->bounding_boxes)
    {
        if (map_label.find(box.Class) == map_label.end())
            continue;
        objects.push_back(DetectedObject(map_label.at(box.Class), box.probability, box.xmin, box.ymin, box.xmax, box.ymax));
    }
    return objects;
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
            time = img0_buf.front()->header.stamp.toSec();
            header = img0_buf.front()->header;
            image0 = get_image_from_msg(img0_buf.front());
            image1 = get_image_from_msg(img1_buf.front());
            if (n++ % 7 == 0 && use_semantic)
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

bool create_env_callback(lvio_fusion_node::CreateEnv::Request &req,
                         lvio_fusion_node::CreateEnv::Response &res)
{
    res.id = Environment::Create();
    return true;
}

sensor_msgs::ImagePtr cv_mat_to_msg(cv::Mat image)
{
    cv_bridge::CvImage cv_image;
    cv_image.encoding = sensor_msgs::image_encodings::MONO8;
    cv_image.image = image;
    return cv_image.toImageMsg();
}

bool step_callback(lvio_fusion_node::Step::Request &req,
                   lvio_fusion_node::Step::Response &res)
{
    Observation obs;
    Weights weights;
    weights.visual = req.visual;
    weights.lidar_ground = req.lidar_ground;
    weights.lidar_surf = req.lidar_surf;
    Environment::Step(req.id, &weights, &obs, &res.reward, (bool *)&res.done);
    res.image = *cv_mat_to_msg(obs.image);
    pcl::toROSMsg(obs.points_ground, res.points_ground);
    pcl::toROSMsg(obs.points_surf, res.points_surf);
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
        out << pair.first << ",";
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
    if (in)
    {
        while (getline(in, line))
        {
            ss << line;
            ss >> time >> x >> y >> z >> qx >> qy >> qz >> qw;
            Environment::ground_truths[time] = SE3d(Quaterniond(qw, qx, qy, qz), Vector3d(x, y, z));
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
        default:
            break;
        }
    }
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
    Agent::SetCore(new Core());
    estimator = Estimator::Ptr(new Estimator(config_file));
    assert(estimator->Init(use_imu, use_lidar, use_navsat, use_loop, use_adapt) == true);

    ROS_WARN("Waiting for images...");

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
    if (use_semantic)
    {
        sub_objects = n.subscribe("/lvio_fusion_node/output_objects", 10, objects_callback);
        pub_detector = n.advertise<sensor_msgs::Image>("/lvio_fusion_node/image_raw", 10);
    }
    if (train)
    {
        clt_init = n.serviceClient<lvio_fusion_node::Init>("lvio_fusion_node/init");
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
