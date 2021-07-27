#ifndef lvio_fusion_FRONTEND_H
#define lvio_fusion_FRONTEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/visual/local_map.h"

namespace lvio_fusion
{

class Backend;

enum class FrontendStatus
{
    BUILDING,
    INITIALIZING,
    TRACKING,
    LOST
};

class Frontend
{
public:
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe, bool remove_moving_points);

    bool AddFrame(Frame::Ptr frame);

    void AddImu(double time, Vector3d acc, Vector3d gyr);

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void UpdateCache();

    void UpdateImu(const Bias &bias_);

    std::mutex mutex;
    FrontendStatus status = FrontendStatus::BUILDING;
    Frame::Ptr current_frame;
    Frame::Ptr last_frame;
    Frame::Ptr last_keyframe;
    LocalMap local_map;
    double init_time = 0;

private:
    bool Track();

    void ResetImu();

    void InitFrame();

    int TrackLastFrame();

    int Relocate(Frame::Ptr base_frame);

    void CreateKeyframe();

    bool InitMap();

    int TriangulateNewPoints();

    void Preintegrate();

    void PredictState();

    // data
    std::weak_ptr<Backend> backend_;
    std::queue<ImuData> imu_buf_;
    imu::Preintegration::Ptr preintegration_last_kf_; // imu pre integration from last key frame
    SE3d last_frame_pose_cache_;
    SE3d relative_i_j_;
    double dt_ = 0;

    // params
    int num_features_init_;
    int num_features_tracking_bad_;
    int num_features_needed_for_keyframe_;
    bool remove_moving_points;
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRONTEND_H
