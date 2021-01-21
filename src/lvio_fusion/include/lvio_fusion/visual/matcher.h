#ifndef lvio_fusion_MATCHER_H
#define lvio_fusion_MATCHER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

// convert opencv points
inline void convert_points(const std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &ps)
{
    ps.resize(kps.size());
    for (int i = 0; i < kps.size(); i++)
    {
        ps[i] = kps[i].pt;
    }
}

// convert opencv points
inline void convert_points(const std::vector<cv::Point2f> &ps, std::vector<cv::KeyPoint> &kps)
{
    kps.resize(ps.size());
    for (int i = 0; i < ps.size(); i++)
    {
        kps[i] = cv::KeyPoint(ps[i], 1);
    }
}

class ORBMatcher
{
public:
    ORBMatcher(int num_features_threshold) : detector_(cv::ORB::create()),
                                             matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")),
                                             num_features_threshold_(num_features_threshold) {}

    // int Search(Frame::Ptr current_frame, Frame::Ptr last_frame, std::vector<cv::Point2f> &kps_current, std::vector<cv::Point2f> &kps_last, std::vector<uchar> &status, std::vector<double> &depths, float thershold);

    void FastFeatureToTrack(cv::Mat &image, std::vector<cv::Point2f> &corners, double minDistance, cv::Mat mask = cv::Mat());

    void Match(cv::Mat &prevImg, cv::Mat &nextImg, std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts, std::vector<uchar> &status);

    int Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame);

    int Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame,
                 std::vector<cv::Point2f> &kps_left, std::vector<cv::Point2f> &kps_right, std::vector<cv::Point2f> &kps_current, std::vector<Vector3d> &pbs);

private:
    cv::Mat ComputeBRIEF(cv::Mat image, std::vector<cv::Point2f> &keypoints);

    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    int num_features_threshold_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MATCHER_H
