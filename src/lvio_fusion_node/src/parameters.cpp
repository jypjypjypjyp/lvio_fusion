#include "parameters.h"

string IMU_TOPIC;
string LIDAR_TOPIC;
string NAVSAT_TOPIC;
string IMAGE0_TOPIC, IMAGE1_TOPIC;
string result_path;
int use_imu, use_lidar, num_of_cam, use_navsat, use_loop, is_semantic;

void read_parameters(string config_file)
{
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
    }

    fsSettings["use_imu"] >> use_imu;
    fsSettings["use_lidar"] >> use_lidar;
    fsSettings["use_navsat"] >> use_navsat;
    fsSettings["use_loop"] >> use_loop;
    fsSettings["num_of_cam"] >> num_of_cam;
    fsSettings["is_semantic"] >> is_semantic;
    fsSettings["result_path"] >> result_path;
    if (num_of_cam == 2)
    {
        fsSettings["image0_topic"] >> IMAGE0_TOPIC;
        fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    }
    if (use_imu)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
    }
    if (use_lidar)
    {
        fsSettings["lidar_topic"] >> LIDAR_TOPIC;
    }
    if (use_navsat)
    {
        fsSettings["navsat_topic"] >> NAVSAT_TOPIC;
    }
    fsSettings.release();
}
