#include "parameters.h"

string IMU_TOPIC;
string LIDAR_TOPIC;
string NAVSAT_TOPIC;
string IMAGE0_TOPIC, IMAGE1_TOPIC;
string NAV_GOAL_TOPIC;//NAVI
string result_path, ground_truth_path;
int use_imu, use_lidar, use_navsat, use_loop, use_semantic, use_adapt, use_navigation, train;//NAVI

void read_parameters(string config_file)
{
    FILE *f = fopen(config_file.c_str(), "r");
    if (f == NULL)
    {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(f);

    cv::FileStorage settings(config_file, cv::FileStorage::READ);
    if (!settings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
    }

    settings["use_imu"] >> use_imu;
    settings["use_lidar"] >> use_lidar;
    settings["use_navsat"] >> use_navsat;
    settings["use_loop"] >> use_loop;
    settings["use_adapt"] >> use_adapt;
    settings["use_navigation"] >> use_navigation;//NAVI
    settings["train"] >> train;
    settings["result_path"] >> result_path;
    settings["ground_truth_path"] >> ground_truth_path;
    settings["image0_topic"] >> IMAGE0_TOPIC;
    settings["image1_topic"] >> IMAGE1_TOPIC;
    if (use_imu)
    {
        settings["imu_topic"] >> IMU_TOPIC;
    }
    if (use_lidar)
    {
        settings["lidar_topic"] >> LIDAR_TOPIC;
    }
    if (use_navsat)
    {
        settings["navsat_topic"] >> NAVSAT_TOPIC;
    }
    if(use_navigation)//NAVI
    {
         settings["nav_goal_topic"] >> NAV_GOAL_TOPIC;
    }
    train = /*use_imu &&*/ use_lidar && train;
    settings.release();
}
