

#ifndef lvio_fusion_FRAME_H
#define lvio_fusion_FRAME_H

#include "lvio_fusion/camera.h"
#include "lvio_fusion/common.h"

namespace lvio_fusion
{

// forward declare
class MapPoint;
class Feature;

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() {}

    Frame(long id, double time, const SE3 &pose, const cv::Mat &left, const cv::Mat &right);

    // set and get pose, thread safe
    SE3& Pose()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pose;
    }

    void SetPose(const SE3 &pose)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        this->pose = pose;
    }

    // set and get velocity, thread safe
    Vector3d& Velocity()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return velocity;
    }

    void SetVelocity(const Vector3d &v)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        this->velocity = v;
    }

    void SetKeyFrame();

    static std::shared_ptr<Frame> CreateFrame();

    unsigned long id = 0;
    unsigned long keyframe_id = 0;
    bool is_keyframe = false;
    double time;
    SE3 pose;
    Vector3d velocity;
    cv::Mat left_image, right_image;
    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left;
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right;

private:
    std::mutex data_mutex_;        
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
