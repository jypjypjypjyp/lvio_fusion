#ifndef lvio_fusion_PARAMETERS_H
#define lvio_fusion_PARAMETERS_H

#include <ros/ros.h>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

extern string IMU_TOPIC;
extern string LIDAR_TOPIC;
extern string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern string result_path;
extern int use_imu;
extern int use_lidar;
extern int use_gnss;
extern int use_rtk;
extern int is_semantic;
extern int num_of_cam;

void read_parameters(std::string config_file);

#endif // lvio_fusion_PARAMETERS_H