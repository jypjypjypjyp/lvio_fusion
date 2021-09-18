#ifndef navigation_GLOBALPLANNER_H
#define navigation_GLOBALPLANNER_H
#include "navigation_node/common.h"
#include "navigation_node/Dstar.h"
namespace navigation_node
{
class Global_planner
{
public:
    typedef std::shared_ptr<Global_planner> Ptr;
    Global_planner(int width_, int height_,double resolution_);

    void SetGoalPose(Vector2d position);
    void SetRobotPose(Vector2d position);
    void SetNewMap(cv::Mat map,int max_x,int max_y,int min_x,int min_y);
    list<Vector2d> GetPath();

    //void SetLocalPlanner(Local_planner::Ptr localplanner ) { localplanner_ = localplanner; }//NAVI

    bool pathupdated=false;
private:
    Dstar *dstar_planner_; // Dstar planner
    cv::Mat costmap;
    Vector2d robot_position;
    Vector2d goal_position;
    list<state> plan_path;
    list<Vector2d> plan_path_;
    int width;
    int height;
    double resolution;
    //Local_planner::Ptr localplanner_;//NAVI 
};

}// namespace navigation_node
#endif // navigation_GLOBALPLANNER_H