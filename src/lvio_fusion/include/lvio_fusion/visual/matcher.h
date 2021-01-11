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
    ORBMatcher() : detector_(cv::ORB::create()) {}

    int Search(Frame::Ptr current_frame, Frame::Ptr last_frame, const float radius);

private:
    void ComputeBRIEF(Frame::Ptr frame);

    bool SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id);

    int Hamming(const BRIEF &a, const BRIEF &b);

    cv::Ptr<cv::Feature2D> detector_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MATCHER_H
