#ifndef navigation_LOCALPLANNER_H
#define navigation_LOCALPLANNER_H
#include "navigation_node/local_planner.h"
namespace navigation_node
{
    Local_planner::Local_planner(int width_, int height_,double resolution_)
    :width(width_), height(height_), resolution(resolution_)
    {
        dwa = DWA::Ptr();
        // process();
    }
    
    void Local_planner::SetPlanPath(std::list<Vector2d> plan_path_)
    {
        plan_path=plan_path_;
    }

    void Local_planner::SetRobotPose(Vector2d robot_position_,  double yaw_)
    {
        Quaterniond q= AngleAxisd(yaw_,Vector3d::UnitZ())*AngleAxisd(0, Vector3d::UnitY())*AngleAxisd(0, Vector3d::UnitX());
        Vector3d t(robot_position_[0], robot_position_[1], 0);
        robot_pose=SE3d(q,t);
        robot_position_changed=true;
    }

    void Local_planner::SetMap(cv::Mat newmap)
    {
            
    }

    void Local_planner::process()
    {
        Vector2d last_goal;
        while(true)
        {
        while(plan_path.size()>0)
        {
            if(robot_position_changed)
            {
                Vector2d goal = plan_path.front();
                if(sqrt((last_goal[0]-robot_pose.translation()[0])*(last_goal[0]-robot_pose.translation()[0])+(last_goal[1]-robot_pose.translation()[1])*(last_goal[1]-robot_pose.translation()[1]))>6)
                {
                    continue;//还没到上一个目标点
                }
                last_goal=goal;
                plan_path.pop_front();
                if (sqrt((goal[0]-robot_pose.translation()[0])*(goal[0]-robot_pose.translation()[0])+(goal[1]-robot_pose.translation()[1])*(goal[1]-robot_pose.translation()[1]))<6)
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
                geometry_msgs::PoseStampedPtr local_goal_msg;
                local_goal_msg->pose.position.x = t.x();
                local_goal_msg->pose.position.y = t.y();
                local_goal_msg->pose.position.z = t.z();
                local_goal_msg->pose.orientation.w = Q.w();
                local_goal_msg->pose.orientation.x = Q.x();
                local_goal_msg->pose.orientation.y = Q.y();
                local_goal_msg->pose.orientation.z = Q.z();
                dwa->set_local_goal(local_goal_msg);
            }
        }
        if(sqrt((last_goal[0]-robot_pose.translation()[0])*(last_goal[0]-robot_pose.translation()[0])+(last_goal[1]-robot_pose.translation()[1])*(last_goal[1]-robot_pose.translation()[1]))>1)
                {
                    continue;//还没到上一个目标点
                }
            //stop robot;
        }
    }


}// namespace navigation_node
#endif // navigation_LOCALPLANNER_H