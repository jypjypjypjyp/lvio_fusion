#ifndef lvio_fusion_MATCHER_H
#define lvio_fusion_MATCHER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/utility.h"

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

// double calculate optical flow
inline int optical_flow(cv::Mat &prevImg, cv::Mat &nextImg,
                        std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                        std::vector<uchar> &status)
{
    if (prevPts.empty())
        return 0;

    cv::Mat err;
    cv::calcOpticalFlowPyrLK(
        prevImg, nextImg, prevPts, nextPts, status, err, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = prevPts;
    cv::calcOpticalFlowPyrLK(
        nextImg, prevImg, nextPts, reverse_pts, reverse_status, err, cv::Size(3, 3), 1,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_success_pts = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        // clang-format off
        if (status[i] && reverse_status[i] && distance(prevPts[i], reverse_pts[i]) <= 0.5
        && nextPts[i].x >= 0 && nextPts[i].x < prevImg.cols
        && nextPts[i].y >= 0 && nextPts[i].y < prevImg.rows)
        // clang-format on
        {
            status[i] = 1;
            num_success_pts++;
        }
        else
            status[i] = 0;
    }
    return num_success_pts;
}

class ORBMatcher
{
public:
    ORBMatcher(int num_features_threshold) : detector_(cv::ORB::create()),
                                             matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")),
                                             num_features_threshold_(num_features_threshold) {}

    void FastFeatureToTrack(cv::Mat &image, std::vector<cv::Point2f> &corners, double minDistance, cv::Mat mask = cv::Mat());

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
