#include "navigation_node/navigation_node.h"

void nav_goal_callback(const geometry_msgs::PoseStamped  &nav_goal_msg)
{
    //LOG(INFO)<<nav_goal_msg.pose.position.x<<" "<<nav_goal_msg.pose.position.y;
    global_planner->SetGoalPose(Vector2d(nav_goal_msg.pose.position.x,nav_goal_msg.pose.position.y));
}

void pose_callback(const geometry_msgs::PoseStamped  &pose_msg)
{
    Vector3d pose3d(pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z);
    Vector2d pose2d(pose3d[0],pose3d[1]);
    //LOG(INFO)<<pose3d[0]<<" "<<pose3d[1];
    global_planner->SetRobotPose(pose2d);
}

void border_callback(const std_msgs::Float64MultiArray::ConstPtr& border)
{
    max_x=int(border->data.at(0));
    max_y=int(border->data.at(2));
    min_x=int(border->data.at(1));
    min_y=int(border->data.at(3));
    //LOG(INFO)<<max_x<<" "<< max_y<<" "<<min_x<<" "<<min_y;
}

void gridmap_callback(const nav_msgs::OccupancyGrid  &gridmap_msg)
{
    cv::Mat map(gridmap_msg.info.width+5,gridmap_msg.info.height+5,CV_32FC1);
    map.setTo(-1);
    for(int row = 0; row<gridmap_msg.info.width;row++)
    {
        for(int col =0; col<gridmap_msg.info.height; col++)
        {
            float val=float(gridmap_msg.data[row*gridmap_msg.info.width+col]);
            if(val!=-1)
            {
                map.at<float>(row,col)=1-val/100;
            }
        }
    }

    global_planner->SetNewMap(map, max_x, max_y, min_x, min_y);
}

void localmap_callbcak(const nav_msgs::OccupancyGrid  &localmap_msg)
{
    
}

void plan_path_timer_callback(const ros::TimerEvent &timer_event)
{
    if(global_planner->pathupdated)
    {
        plan_path.poses.clear();
        list<Vector2d> plan_path_ = global_planner->GetPath();
 
        for( auto point: plan_path_)
        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time(timer_event.current_real.toSec());
            pose_stamped.header.frame_id = "navigation";
            pose_stamped.pose.position.x = point.x();
            pose_stamped.pose.position.y = point.y();
            pose_stamped.pose.position.z = 0;
            plan_path.poses.push_back(pose_stamped);
        }
        plan_path.header.stamp = ros::Time(timer_event.current_real.toSec());
        plan_path.header.frame_id = "navigation";
        LOG(INFO)<<"plan_path_: "<<plan_path_.size();
        pub_plan_path.publish(plan_path);
    }
}

void control_vel_timer_callback(const ros::TimerEvent &timer_event)
{
    
}

// void navi_process()
// {
    
// }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "navigation_node");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    //read_parameters
    string config_file = "read_config_path_is_not_correct";
    if (argc > 1)
    {
        if (argc != 2)
        {
            printf("Please intput: rosrun navigation_node navigation_node [config file] \n"
                   "for example: rosrun navigation_node navigation_node "
                   "~/Projects/lvio-fusion/src/navigation_node/config/kitti.yaml \n");
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
        return 0;
    }
    fclose(f);

    cv::FileStorage settings(config_file, cv::FileStorage::READ);
    if (!settings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
    }
    settings["use_navigation"] >> use_navigation;
    if(use_navigation)
    {
        settings["grid_width"] >> GRID_WIDTH;
        settings["grid_height"] >> GRID_HEIGHT;
        settings["grid_resolution"]>> GRID_RESOLUTION;

        settings["nav_goal_topic"] >> NAV_GOAL_TOPIC;
        settings["pose_topic"] >> POSE_TOPIC;
        settings["border_topic"]>> BORDER_TOPIC;
        settings["gridmap_topic"] >> GRIDMAP_TOPIC;
        settings["use_obstacle_avoidance"] >> use_obstacle_avoidance;
        if(use_obstacle_avoidance)
        {
            settings["localmap_topic"] >> LOCALMAP_TOPIC;
        }
    }
    settings.release();
    ros::Timer plan_path_timer;
    ros::Timer control_vel_timer;

    // if(use_navigation)
    // {

    //     if(use_obstacle_avoidance)
    //     {
            
    //     }
    // }

    //set sub and pub
    if(use_navigation)
    {
        global_planner = navigation_node::Global_planner::Ptr(new navigation_node::Global_planner(
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_RESOLUTION));
        cout << "nav_goal:" << NAV_GOAL_TOPIC << endl;
        sub_nav_goal = n.subscribe(NAV_GOAL_TOPIC, 100, nav_goal_callback);
        cout << "robot_pose:" << POSE_TOPIC << endl;
        sub_pose = n.subscribe(POSE_TOPIC,1,pose_callback);
        cout << "border_pose:" << BORDER_TOPIC << endl;
        sub_border = n.subscribe(BORDER_TOPIC,1,border_callback);
        cout << "gridmap:" << GRIDMAP_TOPIC << endl;
        sub_gridmap = n.subscribe(GRIDMAP_TOPIC,1,gridmap_callback);
        pub_plan_path = n.advertise<nav_msgs::Path>("plan_path", 10);
        plan_path_timer = n.createTimer(ros::Duration(2),plan_path_timer_callback);
        if(use_obstacle_avoidance)
        {
            cout << "localmap:" << LOCALMAP_TOPIC << endl;
            sub_localmap = n.subscribe(LOCALMAP_TOPIC,10,localmap_callbcak);
            pub_control_vel = n.advertise<nav_msgs::Path>("control_vel", 1000);
            control_vel_timer = n.createTimer(ros::Duration(2),control_vel_timer_callback);
        }
    }
    // thread navi_thread{navi_process};
    ros::spin();
    //navi_thread.join();
    return 0;
}
