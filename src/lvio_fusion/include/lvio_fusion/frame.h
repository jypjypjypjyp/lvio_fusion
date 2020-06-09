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

    SE3 &Pose()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        this->pose_ = pose;
    }

    Vector3d &Velocity()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return velocity_;
    }

    void SetVelocity(const Vector3d &v)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        this->velocity_ = v;
    }

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

private:
    //NOTE: semantic map
    LabelType GetLabelType(int x, int y);

    std::mutex data_mutex_;
    SE3 pose_;
    Vector3d velocity_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
