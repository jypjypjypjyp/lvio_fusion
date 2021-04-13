#ifndef lvio_fusion_LOCAL_H
#define lvio_fusion_LOCAL_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

typedef std::bitset<256> BRIEF;

class LocalMap
{
public:
    struct Feature
    {
        typedef std::shared_ptr<Feature> Ptr;

        cv::KeyPoint kp;
        visual::Landmark::Ptr landmark;
        Frame::Ptr frame;
        BRIEF brief;
        bool match = false;
        bool insert = false;

        Feature(Frame::Ptr frame, cv::KeyPoint kp, BRIEF brief) : frame(frame), kp(kp), brief(brief) {}
    };
    typedef std::vector<Feature::Ptr> Features;
    typedef std::vector<std::vector<Feature::Ptr>> FeaturePyramid;

    LocalMap() : detector_(cv::ORB::create(500, 1.2, 4)),
                 matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")),
                 num_levels_(detector_->getNLevels()),
                 scale_factor_(detector_->getScaleFactor())
    {
        double current_factor = 1;
        for (int i = 0; i < num_levels_; i++)
        {
            scale_factors_.push_back(current_factor);
            current_factor *= scale_factor_;
        }
    }

    int Init(Frame::Ptr new_kf);

    void Reset();

    void AddKeyFrame(Frame::Ptr new_kf);

    Features GetLandmarks(Frame::Ptr frame);

    PointRGBCloud GetLocalLandmarks();

    void UpdateCache();

    std::unordered_map<unsigned long, std::pair<double, Vector3d>> position_cache;
    std::unordered_map<double, SE3d> pose_cache;
    double oldest = 0;

private:
    Vector3d ToWorld(Feature::Ptr feature);

    void InsertNewLandmarks(Frame::Ptr frame);

    void GetFeaturePyramid(Frame::Ptr frame, FeaturePyramid &pyramid);

    void GetNewLandmarks(Frame::Ptr frame, FeaturePyramid &pyramid);

    void Triangulate(Frame::Ptr frame, Features &featrues);

    std::vector<double> GetCovisibilityKeyFrames(Frame::Ptr frame);

    void Search(std::vector<double> kfs, Frame::Ptr frame);
    void Search(FeaturePyramid &last_pyramid, SE3d last_pose, FeaturePyramid &current_pyramid, Frame::Ptr frame);
    void Search(FeaturePyramid &last_pyramid, SE3d last_pose, Feature::Ptr feature, Frame::Ptr frame);

    std::mutex mutex_;
    cv::Ptr<cv::ORB> detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::map<double, FeaturePyramid> local_features_;
    std::unordered_map<unsigned long, Feature::Ptr> map_;
    std::vector<double> scale_factors_;
    const int num_levels_;
    const double scale_factor_;
    const int windows_size_ = 3;
};
} // namespace lvio_fusion

#endif //!lvio_fusion_LOCAL_H
