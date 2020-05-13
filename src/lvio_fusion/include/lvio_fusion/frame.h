

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

    unsigned long id_ = 0;         
    unsigned long keyframe_id_ = 0;
    bool is_keyframe_ = false;     
    double time_stamp_;            
    SE3 pose_;                     
    std::mutex pose_mutex_;        
    cv::Mat left_img_, right_img_; 

    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

public: // data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
          const Mat &right);

    // set and get pose, thread safe
    SE3& Pose()
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose)
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    void SetKeyFrame();

    static std::shared_ptr<Frame> CreateFrame();
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRAME_H
