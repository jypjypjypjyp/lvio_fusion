#include "parameters.h"

bool UNEVEN;
std::string IMU_TOPIC;
std::string LIDAR_TOPIC;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
int use_imu, use_lidar, num_of_cam, use_gnss, use_rtk, is_semantic;

void read_parameters(std::string config_file)
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
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["use_imu"] >> use_imu;
    fsSettings["use_lidar"] >> use_lidar;
    fsSettings["use_gnss"] >> use_gnss;
    fsSettings["use_rtk"] >> use_rtk;
    fsSettings["num_of_cam"] >> num_of_cam;
    fsSettings["is_semantic"] >> is_semantic;
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
        fsSettings["lidar_topic"] >> UNEVEN;
        fsSettings["lidar_topic"] >> IMU_TOPIC;
    }
    fsSettings.release();
}
