#ifndef lvio_fusion_FRONTEND_H
#define lvio_fusion_FRONTEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/imu/initializer.h"

namespace lvio_fusion
{

class Backend;

enum class FrontendStatus
{
    BUILDING,
    INITIALIZING,
    TRACKING_GOOD,
    TRACKING_BAD,
    TRACKING_TRY,
    LOST
};

class Frontend
{
public:
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe);

    bool AddFrame(Frame::Ptr frame);

    void AddImu(double time, Vector3d acc, Vector3d gyr);

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }
    //NEWADD
    void UpdateFrameIMU(const Bias &bias_);
    void UpdateFrameIMU();
    void PreintegrateIMU();
    void PredictStateIMU();
    void LocalBA();
    //NEWADDEND
    void UpdateCache();

    FrontendStatus status = FrontendStatus::BUILDING;
    Frame::Ptr current_frame;
    Frame::Ptr last_frame;
    Frame::Ptr current_key_frame;
    SE3d relative_i_j;
    //NEWADD
    imu::Preintegration::Ptr  ImuPreintegratedFromLastKF;
    Frame::Ptr last_key_frame;
    std::list<imuPoint> imuData_buf;
    //NEWADDEND
    std::mutex mutex;

private:
    bool Track();

    bool Reset();

    int TrackLastFrame();

    void CreateKeyframe(bool need_new_features = true);

    bool InitMap();

    int DetectNewFeatures();

    int TriangulateNewPoints();

    // data
    std::weak_ptr<Backend> backend_;
    std::unordered_map<unsigned long, Vector3d> position_cache_;
    SE3d last_frame_pose_cache_;

    // params
    int num_features_;
    int num_features_init_;
    int num_features_tracking_;
    int num_features_tracking_bad_;
    int num_features_needed_for_keyframe_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRONTEND_H
