#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.hpp"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{
class Frontend;

class Initializer
{  
public:
    typedef std::shared_ptr<Initializer> Ptr;

   // bool Initialize(Frames kfs);
    void InitializeIMU(float priorG, float priorA, bool bFIBA);
    void SetMap(Map::Ptr map) { map_ = map; }
    void SetFrontend(Frontend::Ptr  frontend) { frontend_ = frontend; }
    Frontend::Ptr frontend_;
    bool initialized = false;
    
    bool bInitializing=false;

    Eigen::MatrixXd infoInertial;
    int mNumLM;
    int mNumKFCulling;

    float mTinit;  // 用于IMU初始化的时间

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
    
    double mFirstTs;   /// 用于imu初始化第一个可用关键帧的时间

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
    Map::Ptr map_;
    Vector3d g_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
