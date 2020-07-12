#ifndef lvio_fusion_FRONTEND_H
#define lvio_fusion_FRONTEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/sensors/camera.hpp"

namespace lvio_fusion
{

class Backend;

enum class FrontendStatus
{
    INITING,
    TRACKING_GOOD,
    TRACKING_BAD,
    TRACKING_TRY,
    LOST
};

enum Flag
{
    None = 0,
    Mono = 1,
    Stereo = 1 << 1,
    RGBD = 1 << 2,
    IMU = 1 << 3,
    Lidar = 1 << 4,
    GNSS = 1 << 5,
    RTK = 1 << 6,
    Semantic = 1 << 7,
};

class Frontend
{
public:
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    bool AddFrame(Frame::Ptr frame);

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right)
    {
        left_camera_ = left;
        right_camera_ = right;
    }

    int flags = Flag::None;
    FrontendStatus status = FrontendStatus::INITING;
    Frame::Ptr current_frame = nullptr;
    Frame::Ptr last_frame = nullptr;
    SE3d relative_motion;
    std::mutex local_map_mutex;

private:
    bool Track();

    bool Reset();

    int TrackLastFrame();

    int Optimize();

    bool InitFramePoseByPnP();

    void CreateKeyframe();

    bool StereoInit();

    int DetectNewFeatures();

    int FindFeaturesInRight();

    bool InitMap();

    int TriangulateNewPoints();

    // data
    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    Camera::Ptr left_camera_ = nullptr;
    Camera::Ptr right_camera_ = nullptr;

    // params
    int num_features_ = 200;
    int num_features_init_ = 100;
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 8;
    int num_features_needed_for_keyframe_ = 80;
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRONTEND_H
