#include <ros/package.h>
#include <ros/ros.h>
#include <fstream>
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

using namespace std;

navigation_node::Global_planner::Ptr global_planner;

ros::Subscriber  sub_nav_goal,sub_pose, sub_gridmap, sub_localmap, sub_vel;

string NAV_GOAL_TOPIC;
string POSE_TOPIC;
string GRIDMAP_TOPIC;
string LOCALMAP_TOPIC;
string VEL_TOPIC;

int use_navigation;
int use_obstacle_avoidance;

ros::Publisher pub_plan_path , pub_control_vel;
nav_msgs::Path plan_path;
geometry_msgs::Twist control_vel;

int max_x, max_y, min_x, min_y;