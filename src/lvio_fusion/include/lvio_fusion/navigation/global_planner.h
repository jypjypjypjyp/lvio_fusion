#ifndef lvio_fusion_GLOBALPLANNER_H
#define lvio_fusion_GLOBALPLANNER_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/navigation/Dstar.h"
namespace lvio_fusion
{
class Global_planner
{
public:
    typedef std::shared_ptr<Global_planner> Ptr;
    Global_planner(int width_, int height_){
        dstar_planner_ = new Dstar();
        dstar_planner_->init(height_/2, width_/2, height_/2, width_/2); // First initialization
        costmap.create(height_, width_, CV_8SC1);
        costmap.setTo(-1);
    }
    void SetGoalPose(Vector2d position)
    {
        goal_position=position;
        dstar_planner_->updateGoal((goal_position[0]+0.5)/1,(goal_position[1]+0.5)/1);
        dstar_planner_->replan();
        list<state> path =dstar_planner_->getPath();
        
    }
    void SetRobotPose(Vector2d position)
    {
        robot_position=position;
        dstar_planner_->updateStart((robot_position[0]+0.5)/1,(robot_position[1]+0.5)/1);
    }
    void SetGridMap(cv::Mat newmap,int max_x,int max_y,int min_x,int min_y)
    {
            cv::Mat diffmap=newmap-costmap;
            for(int x=min_x;x<max_x;x++)
                for(int y=min_y;y<max_y;y++)
                {
                    if(diffmap.at<int>(x,y)!=0)
                    {
                        dstar_planner_->updateCell(x,y,newmap.at<int>(x,y));
                    }
                }
           costmap=newmap;
    }
private:
    Dstar *dstar_planner_; ///<  Dstar planner
    cv::Mat costmap;
    Vector2d robot_position;
    Vector2d goal_position;
};
} // namespace lvio_fusion
#endif // lvio_fusion_GLOBALPLANNER_H