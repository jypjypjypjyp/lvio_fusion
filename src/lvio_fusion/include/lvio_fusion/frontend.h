
#ifndef lvio_fusion_FRONTEND_H
#define lvio_fusion_FRONTEND_H

#include <opencv2/features2d.hpp>

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class Backend;
class Viewer;

enum class FrontendStatus
{
    INITING,
    TRACKING_GOOD,
    TRACKING_BAD,
    LOST
};

enum Flag
{
    None     = 0,
    Mono     = 1,
    Stereo   = 1<<1,
    RGBD     = 1<<2,
    IMU      = 1<<3,
    Lidar    = 1<<4,
    GNSS     = 1<<5,
    RTK      = 1<<6,
    Semantic = 1<<7,
};

class Frontend
{
public:
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    bool AddFrame(Frame::Ptr frame);

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    int flags = Flag::None;
    FrontendStatus status = FrontendStatus::INITING;
    Frame::Ptr current_frame = nullptr;
    Frame::Ptr last_frame = nullptr;
    Camera::Ptr camera_left = nullptr;
    Camera::Ptr camera_right = nullptr;
    SE3 relative_motion;

    void SetCameras(Camera::Ptr left, Camera::Ptr right)
    {
        camera_left = left;
        camera_right = right;
    }

private:
    bool Track();

    bool Reset();

    int TrackLastFrame();

    int EstimateCurrentPose();

    bool InsertKeyframe();

    bool StereoInit();

    int DetectFeatures();

    int FindFeaturesInRight();

    bool BuildInitMap();

    int TriangulateNewPoints();

    void SetObservationsForKeyFrame();

    // data
    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    int tracking_inliers_ = 0; // inliers, used for testing new keyframes

    // params
    int num_features_ = 200;
    int num_features_init_ = 100;
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 20;
    int num_features_needed_for_keyframe_ = 80;

    // utilities
    cv::Ptr<cv::GFTTDetector> gftt_; // feature detector in opencv
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRONTEND_H
