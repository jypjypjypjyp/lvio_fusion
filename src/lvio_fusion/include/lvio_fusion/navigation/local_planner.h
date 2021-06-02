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
        Local_planner(){}

        void local_goal(std::list<Vector2d> plan_path_){
            
        }

     };
}// namespace lvio_fusion
#endif // lvio_fusion_LOCALPLANNER_H