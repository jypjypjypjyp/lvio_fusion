#ifndef lvio_fusion_OPTIMIZER_H
#define lvio_fusion_OPTIMIZER_H


#include "lvio_fusion/common.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/ceres/imu_error.hpp"

#include <math.h>



namespace lvio_fusion
{
class optimizer
{
public:
  void static InertialOptimization(Map::Ptr pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel=false, bool bGauss=false, float priorG = 1e2, float priorA = 1e6);
    
void static FullInertialBA(Map::Ptr pMap, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0, bool *pbStopFlag=NULL, bool bInit=false, float priorG = 1e2, float priorA=1e6, Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL);

};

} // namespace lvio_fusion

#endif // lvio_fusion_OPTIMIZER_H
