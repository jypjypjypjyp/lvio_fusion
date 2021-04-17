#ifndef lvio_fusion_EXTRACTOR_H
#define lvio_fusion_EXTRACTOR_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

// from orb_slam_2_ros which is a ROS implementation of ORB_SLAM2(https://github.com/appliedAI-Initiative/orb_slam_2_ros)
class QuadTreeNode
{
public:
    QuadTreeNode() : no_more(false) {}

    void DivideNode(QuadTreeNode &n1, QuadTreeNode &n2, QuadTreeNode &n3, QuadTreeNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<QuadTreeNode>::iterator lit;
    bool no_more;
};

class Extractor
{
public:
    Extractor(int nfeatures = 1000, float scaleFactor = 1.2, int nlevels = 8, int iniThFAST = 20, int minThFAST = 7, int patchSize = 31, int edgeThreshold = 31);

    // compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an quad tree.
    void DetectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors);

    const int num_features;
    const double scale_factor;
    const int num_levels;
    const int init_FAST_thershold;
    const int min_FAST_thershold;
    const int patch_size;
    const int half_patch_size;
    const int edge_thershold;

private:
    void ComputePyramid(cv::Mat image);

    void ComputeKeyPointsQuadTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);

    std::vector<cv::KeyPoint> DistributeQuadTree(
        const std::vector<cv::KeyPoint> &vToDistributeKeys,
        const int &minX, const int &maxX,
        const int &minY, const int &maxY,
        const int &nFeatures, const int &level);

    float ICAngle(const Mat &image, Point2f pt);

    std::vector<int> num_features_per_levels_;
    std::vector<int> umax;
    std::vector<float> scale_factor_per_levels_;
    std::vector<float> inv_scale_factor_per_levels_;
    std::vector<float> sigma2_per_levels_;
    std::vector<float> inv_sigma2_per_levels_;
    std::vector<cv::Mat> image_pyramid_;
    cv::Ptr<cv::ORB> orb_;
};

} //namespace lvio_fusion

#endif
