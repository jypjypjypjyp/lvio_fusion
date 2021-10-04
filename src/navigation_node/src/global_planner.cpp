#ifndef lvio_fusion_GLOBALPLANNER_H
#define lvio_fusion_GLOBALPLANNER_H
#include "navigation_node/global_planner.h"
namespace navigation_node
{
    Global_planner::Global_planner(int width_, int height_,double resolution_)
    :width(width_), height(height_), resolution(resolution_)
    {
        dstar_planner_ = new Dstar();
        dstar_planner_->init(height_/2, width_/2, height_/2, width_/2); // First initialization
        costmap.create(height_, width_, CV_32FC1);
        costmap.setTo(-1);
        LOG(INFO)<<"create global_planner";
    }
    void Global_planner::SetGoalPose(Vector2d position)
    {
        goal_position=position;
        dstar_planner_->updateGoal((height/2+goal_position[0]/resolution+0.5)/1,(width/2+goal_position[1]/resolution+0.5)/1);

        dstar_planner_->replan();
        plan_path.clear();
        plan_path_.clear();
        plan_path =dstar_planner_->getPath();
        for(auto state_ :plan_path)
        {
           plan_path_.push_back(Vector2d(resolution*(state_.x-height/2),resolution*(state_.y-width/2)));
        }
        pathupdated=true;
        LOG(INFO)<<"PATH: "<<plan_path_.size();
    }
    void Global_planner::SetRobotPose(Vector2d position)
    {
        robot_position=position;
        dstar_planner_->updateStart((height/2+robot_position[0]/resolution+0.5)/1,(width/2+robot_position[1]/resolution+0.5)/1);
    }
    void Global_planner::SetNewMap(cv::Mat newmap,int max_x,int max_y,int min_x,int min_y)
    {
            for(int x=min_x;x<max_x;x++)
                for(int y=min_y;y<max_y;y++)
                {
                    if(newmap.at<float>(x,y)==-1.0)
                    {
                        continue;
                    }

                    if((newmap.at<float>(x,y)-costmap.at<float>(x,y))!=0)
                    {
                        if(newmap.at<float>(x,y)<0.85)
                        {
                            dstar_planner_->updateCell(y,x,-1);
                            if(newmap.at<float>(x,y)<0.45){
                                dstar_planner_->updateCell(y-1,x-1,-1);
                                dstar_planner_->updateCell(y-1,x,-1);
                                dstar_planner_->updateCell(y-1,x+1,-1);
                                dstar_planner_->updateCell(y,x-1,-1);

                                dstar_planner_->updateCell(y,x+1,-1);
                                dstar_planner_->updateCell(y+1,x-1,-1);
                                dstar_planner_->updateCell(y+1,x,-1);
                                dstar_planner_->updateCell(y+1,x+1,-1);
                            }
                            //LOG(INFO)<<"qian!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "<<x<<"  "<<y<<"  "<<newmap.at<float>(x,y)<<" "<<costmap.at<float>(x,y);
                        }
                        else
                        {
                            dstar_planner_->updateCell(y,x,1);
                            //LOG(INFO)<<"dimian------------------------- "<<x<<"  "<<y<<"  "<<newmap.at<float>(x,y)<<" "<<costmap.at<float>(x,y);
                        }

                    }
                }
            costmap =newmap.clone();
            if(max_x>min_x&&max_y>min_y)
            {
                dstar_planner_->replan();
                plan_path.clear();
                plan_path_.clear();
                plan_path =dstar_planner_->getPath();
                for(auto state_ :plan_path)
                {
                    plan_path_.push_back(Vector2d(resolution*(state_.x-height/2),resolution*(state_.y-width/2)));
                    //LOG(INFO)<<plan_path_.back().transpose();
                }
                pathupdated=true;
                //LOG(INFO)<<"PATH: "<<plan_path_.size();
            }
    }

   list<Vector2d> Global_planner::GetPath()
   {
       pathupdated=false;
       return plan_path_;
   }
}// namespace navigation_node
#endif // lvio_fusion_GLOBALPLANNER_H