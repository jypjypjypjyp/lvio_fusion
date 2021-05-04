#ifndef lvio_fusion_LOCAL_H
#define lvio_fusion_LOCAL_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/extractor.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

typedef std::vector<visual::Feature::Ptr> Level;
typedef std::vector<Level> Pyramid;

extern cv::Mat img_track;

class LocalMap
{
public:
    LocalMap(int num_features) : extractor_(num_features),
                                 matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")),
                                 num_levels_(extractor_.num_levels)
    {
        double current_factor = 1;
        for (int i = 0; i < num_levels_; i++)
        {
            scale_factors_.push_back(current_factor);
            current_factor *= extractor_.scale_factor;
        }
    }

    int Init(Frame::Ptr new_kf);

    void Reset();

    void AddKeyFrame(Frame::Ptr new_kf);

    Level GetFeatures(double time);

    PointRGBCloud GetLocalLandmarks();

    void UpdateCache();

    std::unordered_map<unsigned long, Vector3d> position_cache;
    std::unordered_map<double, SE3d> pose_cache;
    visual::Landmarks landmarks;

private:
    Vector3d ToWorld(visual::Feature::Ptr feature);

    void InsertNewLandmarks(Frame::Ptr frame);

    void GetFeaturePyramid(Frame::Ptr frame, Pyramid &pyramid);

    void GetNewLandmarks(Frame::Ptr frame, Pyramid &pyramid);

    void Triangulate(Frame::Ptr frame, Level &featrues);

    std::vector<double> GetCovisibilityKeyFrames(Frame::Ptr frame);

    void Search(std::vector<double> kfs, Frame::Ptr frame);
    void Search(Pyramid &last_pyramid, SE3d last_pose, Pyramid &current_pyramid, Frame::Ptr frame);
    void Search(Pyramid &last_pyramid, SE3d last_pose, visual::Feature::Ptr feature, Frame::Ptr frame);

    std::mutex mutex_;
    Extractor extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::map<double, Pyramid> local_features_;
    std::vector<double> scale_factors_;

    const int num_levels_;
    const int windows_size_ = 3;
};
} // namespace lvio_fusion

#endif //!lvio_fusion_LOCAL_H
