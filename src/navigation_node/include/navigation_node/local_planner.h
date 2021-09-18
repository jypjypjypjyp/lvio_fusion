#ifndef navigation_node_LOCALPLANNER_H
#define navigation_node_LOCALPLANNER_H
#include "navigation_node/common.h"
#include "navigation_node/DWA.h"
namespace navigation_node
{
 class Local_planner
    {
    public:
        typedef std::shared_ptr<Local_planner> Ptr;
        Local_planner(int width_, int height_,double resolution_);
        void SetPlanPath(std::list<Vector2d> plan_path_);
        void SetRobotPose(Vector2d robot_position_,  double yaw_);
        void SetMap(cv::Mat newmap);
        void process();

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
}// namespace navigation_node
#endif // navigation_node_LOCALPLANNER_H