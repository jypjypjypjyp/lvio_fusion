#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/adapt/observation.h"
#include "lvio_fusion/adapt/weights.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/loop/loop.h"
#include "lvio_fusion/navsat/feature.h"
#include "lvio_fusion/semantic/detected_object.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() {}

    void AddFeature(visual::Feature::Ptr feature);

    void RemoveFeature(visual::Feature::Ptr feature);

    Observation GetObservation();

    void Clear();

    //NOTE: semantic map
    void UpdateLabel();

    static Frame::Ptr Create();

    static unsigned long current_frame_id;
    unsigned long id;
    double time;
    cv::Mat image_left, image_right;
    std::vector<DetectedObject> objects;
    visual::Features features_left;          // extracted features in left image
    visual::Features features_right;         // corresponding features in right image, only for this frame
    lidar::Feature::Ptr feature_lidar;       // extracted features in lidar point cloud
    imu::Preintegration::Ptr preintegration; // imu pre integration
    imu::Preintegration::Ptr preintegrationFrame; // imu pre integration from last frame NEWADD
    navsat::Feature::Ptr feature_navsat;     // navsat point
    cv::Mat descriptors;                     // orb descriptors
    loop::LoopClosure::Ptr loop_closure;     // loop closure
    Weights weights;
    SE3d pose;

   //IMU
    Frame::Ptr last_keyframe;
    Vector3d Vw;// IMU linear velocity
    Bias ImuBias;
    bool bImu=false;  //是否经过imu尺度优化

    Vector3d GetGyroBias();
    Vector3d GetAccBias();
    Matrix3d  GetImuRotation();
    Vector3d  GetImuPosition();
    void SetVelocity(const Vector3d  &Vw_);
    Bias GetImuBias();
    Vector3d GetVelocity();
    void SetNewBias(const Bias &bias_);
    void SetPose(const Matrix3d &Rwb_,const Vector3d  &twb_);
   //IMUEND
private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);
};

typedef std::map<double, Frame::Ptr> Frames;

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
