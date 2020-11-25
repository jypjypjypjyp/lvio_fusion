#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/semantic/detected_object.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/loop/loop_constraint.h"

namespace lvio_fusion
{

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame():mTcw(Matrix4d::Zero()),mImuBias(Bias(0,0,0,0,0,0)),mVw(Vector3d::Zero()){ }

    void AddFeature(visual::Feature::Ptr feature);

    void RemoveFeature(visual::Feature::Ptr feature);

    //NOTE: semantic map
    void UpdateLabel();

    static Frame::Ptr Create();

    static unsigned long current_frame_id;
    unsigned long id;
    double time;
    cv::Mat image_left, image_right;
    std::vector<DetectedObject> objects;
    visual::Features features_left;             // extracted features in left image
    visual::Features features_right;            // corresponding features in right image, only for this frame
    lidar::Feature::Ptr feature_lidar;          // extracted features in lidar point cloud
    imu::Preintegration::Ptr preintegration;    // imu pre integration
    cv::Mat descriptors;                        // orb descriptors
    loop::LoopConstraint::Ptr loop_constraint;  // loop constraint
    SE3d pose;

    Frame::Ptr mpLastKeyFrame;

//IMU
    Calib calib_;
    // Rotation, translation and camera center
    Matrix3d mRcw;
    Vector3d mtcw;
    Matrix3d mRwc;
    Vector3d mOw;

    // IMU linear velocity
    Vector3d mVw;

    // Camera pose.
    Matrix4d  mTcw;  //? world to camera 

    Bias mImuBias;
    bool bImu;  //是否经过imu尺度优化
    // IMU position
    Vector3d Owb;

    Vector3d GetGyroBias();
    Vector3d GetAccBias();
    Matrix3d  GetImuRotation();
    Vector3d  GetImuPosition();
    void SetVelocity(const Vector3d  &Vw_);
    Bias GetImuBias();
    Matrix4d GetPoseInverse();
    Vector3d GetVelocity();
    void SetPose(const Matrix3d Rcw,const Vector3d tcw);
    void SetNewBias(const Bias &b);
    void SetImuPoseVelocity(const Matrix3d &Rwb,const Vector3d &twb, const Vector3d &Vwb);
    void UpdatePoseMatrices();

private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);
};

typedef std::map<double, Frame::Ptr> Frames;

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
