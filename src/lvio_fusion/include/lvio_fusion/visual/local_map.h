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
    struct Point
    {
        typedef std::shared_ptr<Point> Ptr;

        cv::KeyPoint kp;
        visual::Landmark::Ptr landmark;
        Frame::Ptr frame;
        BRIEF brief;
        bool match = false;
        bool insert = false;

        Point(Frame::Ptr frame, cv::KeyPoint kp, BRIEF brief) : frame(frame), kp(kp), brief(brief) {}
    };
    typedef std::vector<Point::Ptr> Points;
    typedef std::vector<std::vector<Point::Ptr>> Pyramid;

    LocalMap() : detector_(cv::ORB::create(250, 1.2, 4)),
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

    Points GetLandmarks(Frame::Ptr frame);

    PointRGBCloud GetLocalLandmarks();

    void UpdateCache();

    std::unordered_map<unsigned long, std::pair<double, Vector3d>> position_cache;
    std::unordered_map<double, SE3d> pose_cache;
    double oldest = 0;

private:
    Vector3d ToWorld(Point::Ptr feature);

    void InsertNewLandmarks(Frame::Ptr frame);

    void GetFeaturePyramid(Frame::Ptr frame, Pyramid &pyramid);

    void GetNewLandmarks(Frame::Ptr frame, Pyramid &pyramid);

    void Triangulate(Frame::Ptr frame, Points &featrues);

    std::vector<double> GetCovisibilityKeyFrames(Frame::Ptr frame);

    void Search(std::vector<double> kfs, Frame::Ptr frame);
    void Search(Pyramid &last_pyramid, SE3d last_pose, Pyramid &current_pyramid, Frame::Ptr frame);
    void Search(Pyramid &last_pyramid, SE3d last_pose, Point::Ptr feature, Frame::Ptr frame);

    std::mutex mutex_;
    cv::Ptr<cv::ORB> detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::map<double, Pyramid> local_features_;
    std::unordered_map<unsigned long, Point::Ptr> map_;
    std::vector<double> scale_factors_;
    const int num_levels_;
    const double scale_factor_;
    const int windows_size_ = 3;
};
} // namespace lvio_fusion

#endif //!lvio_fusion_LOCAL_H
