#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/camera.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/semantic/detected_object.h"

namespace lvio_fusion
{

typedef std::list<Feature::Ptr> Features;

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() {}

    Frame(long id, double time, const SE3 &pose, const cv::Mat &left, const cv::Mat &right);

    void SetKeyFrame();

    void AddFeature(Feature::Ptr feature);

    void RemoveFeature(Feature::Ptr feature);

    //NOTE: semantic map
    void UpdateLabel();

    static Frame::Ptr CreateFrame();

    unsigned long id = -1;
    double time;
    cv::Mat left_image, right_image;
    std::vector<DetectedObject> objects;
    // extracted features in left image
    Features left_features;
    // corresponding features in right image, set to nullptr if no corresponding
    Features right_features;
    SE3 pose;
    Vector3d velocity;

private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
