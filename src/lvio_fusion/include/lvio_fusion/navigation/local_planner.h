#ifndef lvio_fusion_LOCALPLANNER_H
#define lvio_fusion_LOCALPLANNER_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/navigation/DWA.h"
#include <list>

namespace lvio_fusion
{

    class Local_planner
    {
    public:
        typedef std::shared_ptr<Local_planner> Ptr;
        Local_planner(int width_, int height_,double resolution_)
        :width(width_), height(height_), resolution(resolution_)
        {
            dwa = DWA::Ptr();
            process();
        }

        void SetPlanPath(std::list<Vector2d> plan_path_)//global_planner
        {
            plan_path=plan_path_;
        }

        void SetRobotPose(Vector2d robot_position_,  double yaw_)//frontend
        {
            Quaterniond q= AngleAxisd(yaw_,Vector3d::UnitZ())*AngleAxisd(0, Vector3d::UnitY())*AngleAxisd(0, Vector3d::UnitX());
            Vector3d t(robot_position_[0], robot_position_[1], 0);
            robot_pose=SE3d(q,t);
            robot_position_changed=true;
        }

        void SetMap(cv::Mat newmap)//gridmap
        {
            
        }

        void process()
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

    private:
        std::list<Vector2d> plan_path;
        SE3d robot_pose;
        double final_yaw;
        sensor_msgs::LaserScan scan;
        geometry_msgs::Twist current_velocity;
        bool robot_position_changed=false;
        DWA::Ptr dwa;
        double PREDICT_TIME;
        int width;
        int height;
        double resolution;
    };
}// namespace lvio_fusion
#endif // lvio_fusion_LOCALPLANNER_H