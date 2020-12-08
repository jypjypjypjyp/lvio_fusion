#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{
class Frontend;

class Initializer
{  
public:
    typedef std::shared_ptr<Initializer> Ptr;
    
   // bool Initialize(Frames kfs);
    void InitializeIMU( bool bFIBA);
    void SetFrontend(std::shared_ptr<Frontend>  frontend) { frontend_ = frontend; }
    std::weak_ptr<Frontend> frontend_;
    bool initialized = false;
    bool bimu=false;//是否经过imu尺度优化
  //  bool bInitializing=false;

    Eigen::MatrixXd infoInertial;
    int mNumLM;
    int mNumKFCulling;

    double mTinit;  // 用于IMU初始化的时间

    int countRefinement;
    int num_frames = 10;
    
    Eigen::MatrixXd mcovInertial;
    Eigen::Matrix3d mRwg;       /// 重力方向
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    double mInitTime;   /// 未使用
    double mCostTime;
    bool mbNewInit;     /// 当前地图的imu初始化完成
    unsigned int mInitSect;
    unsigned int mIdxInit;  /// map_imu初始化的index
    unsigned int mnKFs; /// 初始化imu时的KF num
    double mFirstTs;   /// 用于imu初始化第一个可用关键帧的时间
    int mnMatchesInliers;
    

   /* class Frame
    {
    public:
        imu::Preintegration::Ptr preintegration;
        Matrix3d R;
        Vector3d T;
        Vector3d Ba, Bg;
    };
*/
private:
   // bool VisualInitialAlign();

   // void SolveGyroscopeBias(std::vector<Initializer::Frame> &frames);

    // void RefineGravity(VectorXd &x);

    // bool LinearAlignment(VectorXd &x);

    // bool VisualIMUAlignment(VectorXd &x);
    Vector3d g_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
