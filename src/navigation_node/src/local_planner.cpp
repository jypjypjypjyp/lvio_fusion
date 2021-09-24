#ifndef navigation_LOCALPLANNER_H
#define navigation_LOCALPLANNER_H
#include "navigation_node/local_planner.h"
namespace navigation_node
{
    Local_planner::Local_planner()
    {
        //dwa = DWA::Ptr();
        // process();
        local_goal_updated=false;
    }
    
    void Local_planner::SetPlanPath(std::list<Vector2d> plan_path_)
    {
        //LOG(INFO)<<"plan_path_"<<plan_path_.size();
        plan_path.clear();
        for(auto i:plan_path_)
        {
            plan_path.push_back(i);
        }
    }

    void Local_planner::SetRobotPose(Vector2d robot_position_,  double yaw_)
    {
        //LOG(INFO)<<"robot_position_"<<robot_position_(0)<<" "<<robot_position_(1);
        Quaterniond q= AngleAxisd(yaw_,Vector3d::UnitZ())*AngleAxisd(0, Vector3d::UnitY())*AngleAxisd(0, Vector3d::UnitX());
        Vector3d t(robot_position_[0], robot_position_[1], 0);
        robot_pose=SE3d(q,t);
        robot_position_changed=true;
    }

    void Local_planner::SetOdom(const nav_msgs::OdometryConstPtr& odom_msg)
    {
        LOG(INFO)<<"odom:"<<odom_msg->twist.twist.linear.x<<" "<<odom_msg->twist.twist.linear.y<<" "<<odom_msg->twist.twist.linear.z;
        //dwa->set_odom(odom_msg);
    }

    void Local_planner::SetMap(const nav_msgs::OccupancyGridConstPtr& newmap)
    {
        //dwa->set_local_map(newmap);
    }

    void Local_planner::process()
    {
        Vector2d last_goal;
        bool first=true;
        LOG(INFO)<<"porcess333";
        while(true)
        {
             if(plan_path.size()>0)
        std::cout<<"plan_path.size()"<<plan_path.size()<<std::endl;
        while(plan_path.size()>0)
        {
            if(robot_position_changed)
            {
                Vector2d goal = plan_path.front();
                //LOG(INFO)<<(sqrt((last_goal[0]-robot_pose.translation()[0])*(last_goal[0]-robot_pose.translation()[0])+(last_goal[1]-robot_pose.translation()[1])*(last_goal[1]-robot_pose.translation()[1]))>6);
                if(!first)
                {
                    if(sqrt((last_goal[0]-robot_pose.translation()[0])*(last_goal[0]-robot_pose.translation()[0])+(last_goal[1]-robot_pose.translation()[1])*(last_goal[1]-robot_pose.translation()[1]))>6)
                    {
                        continue;//还没到上一个目标点
                    }
                }
                else
                {
                    first=false;
                }

                last_goal=goal;
                plan_path.pop_front();
                if (sqrt((goal[0]-robot_pose.translation()[0])*(goal[0]-robot_pose.translation()[0])+(goal[1]-robot_pose.translation()[1])*(goal[1]-robot_pose.translation()[1]))<=6)
                {
                    continue;//已经接近当前目标点；
                }
                double goal_yaw=0;
                if(plan_path.size()>0)
                {
                    Vector2d next_goal = plan_path.front();
                    Vector2d vec=next_goal-goal;
                    goal_yaw = atan2(vec[1], vec[0]);
                } 
                else
                {
                    goal_yaw = final_yaw;
                }
                Quaterniond Q= AngleAxisd(goal_yaw, Vector3d::UnitZ())*AngleAxisd(0, Vector3d::UnitY())*AngleAxisd(0, Vector3d::UnitX());
                Vector3d t(goal[0], goal[1], 0);
                SE3d goal_pose=SE3d(Q, t);
                SE3d tans_pose = robot_pose.inverse()*goal_pose;
               
                local_goal_msg->pose.position.x = t.x();
                local_goal_msg->pose.position.y = t.y();
                local_goal_msg->pose.position.z = t.z();
                local_goal_msg->pose.orientation.w = Q.w();
                local_goal_msg->pose.orientation.x = Q.x();
                local_goal_msg->pose.orientation.y = Q.y();
                local_goal_msg->pose.orientation.z = Q.z();
                local_goal_updated=true;
                LOG(INFO)<<"local_goal_msg "<<local_goal_msg->pose.position.x<<" "<<local_goal_msg->pose.position.y<<" "<<local_goal_msg->pose.position.z;
                //dwa->set_local_goal(local_goal_msg);
            }
        }
        if(sqrt((last_goal[0]-robot_pose.translation()[0])*(last_goal[0]-robot_pose.translation()[0])+(last_goal[1]-robot_pose.translation()[1])*(last_goal[1]-robot_pose.translation()[1]))>3)
        {
            continue;//还没到上一个目标点
        }
            //stop robot;
        }
    }
}// namespace navigation_node
#endif // navigation_LOCALPLANNER_H