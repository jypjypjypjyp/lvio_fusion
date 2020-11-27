#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/semantic/detected_object.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/loop/loop_constraint.h"
#include "lvio_fusion/adapt/weights.h"

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
    imu::Preintegration::Ptr preintegration;    // imu pre integration
    cv::Mat descriptors;                        // orb descriptors
    loop::LoopConstraint::Ptr loop_constraint;  // loop constraint
    Weights weights;
    SE3d pose;

private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);
};

typedef std::map<double, Frame::Ptr> Frames;

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
