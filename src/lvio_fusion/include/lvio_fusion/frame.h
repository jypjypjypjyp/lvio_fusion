#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

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

    static Frame::Ptr Create();

    unsigned long id;
    double time;
    cv::Mat image_left, image_right;
    std::vector<DetectedObject> objects;
    // extracted features in left image
    Features features_left;
    // corresponding features in right image, only for this frame
    Features features_right;
    SE3d pose;

private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
