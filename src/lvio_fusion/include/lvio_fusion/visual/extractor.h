#ifndef lvio_fusion_EXTRACTOR_H
#define lvio_fusion_EXTRACTOR_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

// from ORB_SLAM2
class QuadTreeNode
{
public:
    QuadTreeNode() : no_more(false) {}

    void DivideNode(QuadTreeNode &n1, QuadTreeNode &n2, QuadTreeNode &n3, QuadTreeNode &n4);

    std::vector<cv::KeyPoint> kps;
    cv::Point2i UL, UR, BL, BR;
    std::list<QuadTreeNode>::iterator iter;
    bool no_more;
};

class Extractor
{
public:
    Extractor(int nfeatures = 500, float scaleFactor = 1.2, int nlevels = 4, int iniThFAST = 14, int minThFAST = 7, int patchSize = 31, int edgeThreshold = 31);

    // detect the ORB features on an image.
    // ORB are dispersed on the image using an quad tree.
    void Detect(cv::Mat image, std::vector<std::vector<cv::KeyPoint>> &keypoints);

    // compute the ORB descriptors after detecting.
    cv::Mat Compute(std::vector<std::vector<cv::KeyPoint>> &keypoints);

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

    void ComputeKeyPointsQuadTree(std::vector<std::vector<cv::KeyPoint>> &keypoints);

    std::vector<cv::KeyPoint> DistributeQuadTree(
        const std::vector<cv::KeyPoint> &vToDistributeKeys,
        const int &minX, const int &maxX,
        const int &minY, const int &maxY,
        const int &nFeatures, const int &level);

    void ComputeOrientation(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints);
    float ICAngle(const cv::Mat &image, cv::Point2f pt);

    std::vector<int> umax_;
    std::vector<float> scale_factor_per_levels_;
    std::vector<float> inv_scale_factor_per_levels_;
    std::vector<float> sigma2_per_levels_;
    std::vector<float> inv_sigma2_per_levels_;
    std::vector<int> num_desired_features_;
    std::vector<cv::Mat> image_pyramid_;
};

} //namespace lvio_fusion

#endif
