#ifndef lvio_fusion_MATCHER_H
#define lvio_fusion_MATCHER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

class ORBMatcher
{
public:
    ORBMatcher(int num_features_threshold) : detector_(cv::ORB::create()),
                                             matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")),
                                             num_features_threshold_(num_features_threshold) {}

    void Match(cv::Mat &prevImg, cv::Mat &nextImg, std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts, std::vector<uchar> &status);

    int Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame, SE3d &relative_o_c);

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
