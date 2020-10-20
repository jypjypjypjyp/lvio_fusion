#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/semantic/detected_object.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"
#include <DBoW3/FeatureVector.h>

namespace lvio_fusion
{

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() {}

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
    imu::Preintegration::Ptr preintegration;
    cv::Mat descriptors;
    Frame::Ptr loop;
    SE3d pose;
    Vector3d velocity;                          // velocity



//NEWADD
//IMU

    Bias mImuBias;

    // IMU position
    cv::Mat Owb;

    // Velocity (Only used for inertial SLAM)
    cv::Mat Vw;
cv::Mat GetGyroBias();
cv::Mat GetAccBias();
   cv::Mat   GetImuRotation();
   cv::Mat   GetImuPosition();
   void SetVelocity(const cv::Mat &Vw);
//NEWADDEND


private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);



};

typedef std::map<double, Frame::Ptr> Frames;

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
