#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/camera.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/semantic/detected_object.h"

namespace lvio_fusion
{

typedef std::map<unsigned long ,Feature::Ptr> Features;

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() {}

    void AddFeature(Feature::Ptr feature);

    void RemoveFeature(Feature::Ptr feature);

    //NOTE: semantic map
    void UpdateLabel();

    static Frame::Ptr CreateFrame();

    unsigned long id;
    double time;
    cv::Mat left_image, right_image;
    std::vector<DetectedObject> objects;
    // extracted features in left image
    Features left_features;
    // corresponding features in right image, only for this frame
    Features right_features;
    SE3d pose;
    Vector3d velocity;

private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
