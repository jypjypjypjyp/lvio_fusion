#ifndef lvio_fusion_MATCHER_H
#define lvio_fusion_MATCHER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

typedef std::bitset<256> BRIEF;

inline cv::Mat brief2mat(BRIEF &brief)
{
    return cv::Mat(1, 32, CV_8U, reinterpret_cast<uchar *>(&brief));
}

inline BRIEF mat2brief(const cv::Mat &mat)
{
    BRIEF brief;
    memcpy(&brief, mat.data, 32);
    return brief;
}

inline std::map<unsigned long, BRIEF> mat2briefs(Frame::Ptr frame)
{
    std::map<unsigned long, BRIEF> briefs;
    int i = 0;
    for (auto &pair_feature : frame->features_left)
    {
        briefs[pair_feature.first] = mat2brief(frame->descriptors.row(i));
        i++;
    }
    return briefs;
}

class ORBMatcher
{
public:
    ORBMatcher(int num_features_threshold) : detector_(cv::ORB::create()),
                                             matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")),
                                             num_features_threshold_(num_features_threshold) {}

    // int Search(Frame::Ptr current_frame, Frame::Ptr last_frame, std::vector<cv::Point2f> &kps_current, std::vector<cv::Point2f> &kps_last, std::vector<uchar> &status, std::vector<double> &depths, float thershold);

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
