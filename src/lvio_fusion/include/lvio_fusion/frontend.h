#ifndef lvio_fusion_FRONTEND_H
#define lvio_fusion_FRONTEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/visual/matcher.h"

namespace lvio_fusion
{

class Backend;

enum class FrontendStatus
{
    BUILDING,
    INITIALIZING,
    TRACKING_GOOD,
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

    void UpdateCache();

    FrontendStatus status = FrontendStatus::BUILDING;
    Frame::Ptr current_frame;
    Frame::Ptr last_frame;
    Frame::Ptr last_key_frame;
    SE3d relative_i_j;
    std::mutex mutex;

private:
    bool Track();

    bool Reset();

    int TrackLastFrame(Frame::Ptr base_frame);

    int Relocate(Frame::Ptr base_frame);

    void CreateKeyframe();

    bool InitMap();

    int DetectNewFeatures();

    int TriangulateNewPoints();

    // data
    std::weak_ptr<Backend> backend_;
    ORBMatcher matcher_; 
    std::unordered_map<unsigned long, Vector3d> position_cache_;
    SE3d last_frame_pose_cache_;

    // params
    int num_features_;
    int num_features_init_;
    int num_features_tracking_bad_;
    int num_features_needed_for_keyframe_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRONTEND_H
