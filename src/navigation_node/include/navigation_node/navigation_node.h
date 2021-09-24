#include <ros/package.h>
#include <ros/ros.h>
#include <fstream>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>

#include "navigation_node/common.h"
#include "navigation_node/global_planner.h"
#include "navigation_node/local_planner.h"

using namespace std;

navigation_node::Global_planner::Ptr global_planner;
navigation_node::Local_planner::Ptr local_planner;

ros::Subscriber  sub_nav_goal,sub_pose, sub_border, sub_gridmap,sub_odom ,sub_localmap;

int GRID_WIDTH;
int GRID_HEIGHT;
double GRID_RESOLUTION;

string NAV_GOAL_TOPIC;
string POSE_TOPIC;
string BORDER_TOPIC;
string GRIDMAP_TOPIC;
string ODOM_TOPIC;
string LOCALMAP_TOPIC;

int use_navigation;
int use_obstacle_avoidance;

ros::Publisher pub_plan_path , pub_control_vel, pub_local_goal;
nav_msgs::Path plan_path;
geometry_msgs::Twist control_vel;

int max_x, max_y, min_x, min_y;