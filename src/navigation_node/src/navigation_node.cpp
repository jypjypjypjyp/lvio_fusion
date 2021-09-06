#include "navigation_node/navigation_node.h"
using namespace std;

ros::Subscriber  sub_nav_goal,sub_tf, sub_gridmap, sub_localmap, sub_vel;
ros::Publisher pub_plan_path;
string NAV_GOAL_TOPIC;
int use_navigation;

void nav_goal_callback(const geometry_msgs::PoseStamped  &nav_goal_msg)
{
    
}




int main(int argc, char **argv)
{
    ros::init(argc, argv, "navigation_node");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    string config_file = "read_config_path_is_not_correct";
    if (argc > 1)
    {
        if (argc != 2)
        {
            printf("Please intput: rosrun navigation_node navigation_node [config file] \n"
                   "for example: rosrun navigation_node navigation_node "
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
    settings["use_navigation"] >> use_navigation;
    if(use_navigation)//NAVI
    {
         settings["nav_goal_topic"] >> NAV_GOAL_TOPIC;
    }
    sub_nav_goal = n.subscribe(NAV_GOAL_TOPIC, 100, nav_goal_callback);
    return 0;
}