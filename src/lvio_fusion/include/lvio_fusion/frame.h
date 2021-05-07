#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/adapt/observation.h"
#include "lvio_fusion/adapt/weights.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/loop/loop.h"
#include "lvio_fusion/navsat/feature.h"
#include "lvio_fusion/visual/feature.h"

namespace lvio_fusion
{

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame();

    void AddFeature(visual::Feature::Ptr feature);

    void RemoveFeature(visual::Feature::Ptr feature);

    Observation GetObservation();

    void Clear();

    static Frame::Ptr Create();

    static unsigned long current_frame_id;
    unsigned long id;
    double time;
    Frame::Ptr last_keyframe;
    cv::Mat image_left, image_right;
    visual::Features features_left;               // extracted features in left image
    visual::Features features_right;              // corresponding features in right image, only for this frame
    lidar::Feature::Ptr feature_lidar;            // extracted features in lidar point cloud
    imu::Preintegration::Ptr preintegration;      // imu pre integration from last key frame
    imu::Preintegration::Ptr preintegration_last; // imu pre integration from last frame
    navsat::Feature::Ptr feature_navsat;          // navsat point
    cv::Mat descriptors;                          // orb descriptors
    loop::LoopClosure::Ptr loop_closure;          // loop closure
    Weights weights;                              // weights of different factors
    SE3d pose;

    Vector3d GetGyroBias();
    Vector3d GetAccBias();
    Matrix3d GetRotation();
    Vector3d GetPosition();
    void SetVelocity(const Vector3d &Vw_);
    Bias bias;
    void SetImuBias(const Bias &bias_);
    void SetPose(const Matrix3d &Rwb_, const Vector3d &twb_);

    Vector3d Vw;              // Imu linear velocity
    Bias bias;             // Imu bias
    bool is_imu_good = false; // can be used in Imu optimization?
};

typedef std::map<double, Frame::Ptr> Frames;

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
