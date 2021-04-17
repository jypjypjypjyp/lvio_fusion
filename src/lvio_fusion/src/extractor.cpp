#include "lvio_fusion/visual/extractor.h"

using namespace cv;
using namespace std;

namespace lvio_fusion
{

Extractor::Extractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST, int patchSize, int edgeThreshold)
    : num_features(nfeatures), scale_factor(scaleFactor), num_levels(nlevels),
      init_FAST_thershold(iniThFAST), min_FAST_thershold(minThFAST),
      patch_size(patchSize), half_patch_size(patchSize / 2), edge_thershold(edgeThreshold)
{
    orb_ = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, 0, 2, cv::ORB::FAST_SCORE, patchSize, iniThFAST);

    scale_factor_per_levels_.resize(num_levels);
    sigma2_per_levels_.resize(num_levels);
    scale_factor_per_levels_[0] = 1.0f;
    sigma2_per_levels_[0] = 1.0f;
    for (int i = 1; i < num_levels; i++)
    {
        scale_factor_per_levels_[i] = scale_factor_per_levels_[i - 1] * scale_factor;
        sigma2_per_levels_[i] = scale_factor_per_levels_[i] * scale_factor_per_levels_[i];
    }

    inv_scale_factor_per_levels_.resize(num_levels);
    inv_sigma2_per_levels_.resize(num_levels);
    for (int i = 0; i < num_levels; i++)
    {
        inv_scale_factor_per_levels_[i] = 1.0f / scale_factor_per_levels_[i];
        inv_sigma2_per_levels_[i] = 1.0f / sigma2_per_levels_[i];
    }

    image_pyramid_.resize(num_levels);

    num_features_per_levels_.resize(num_levels);
    float factor = 1.0f / scale_factor;
    float nDesiredFeaturesPerScale = num_features * (1 - factor) / (1 - (float)pow((double)factor, (double)num_levels));

    int sumFeatures = 0;
    for (int level = 0; level < num_levels - 1; level++)
    {
        num_features_per_levels_[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += num_features_per_levels_[level];
        nDesiredFeaturesPerScale *= factor;
    }
    num_features_per_levels_[num_levels - 1] = std::max(num_features - sumFeatures, 0);

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(half_patch_size + 1);

    int v, v0, vmax = cvFloor(half_patch_size * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(half_patch_size * sqrt(2.f) / 2);
    const double hp2 = half_patch_size * half_patch_size;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = half_patch_size, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

inline float ICAngle(const Mat &image, Point2f pt)
{
    int m_01 = 0, m_10 = 0;

    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -half_patch_size; u <= half_patch_size; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= half_patch_size; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = umax[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

inline void compute_orientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax)
{
    for (auto keypoint = keypoints.begin(), keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = ICAngle(image, keypoint->pt);
    }
}

void QuadTreeNode::DivideNode(QuadTreeNode &n1, QuadTreeNode &n2, QuadTreeNode &n3, QuadTreeNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.no_more = true;
    if (n2.vKeys.size() == 1)
        n2.no_more = true;
    if (n3.vKeys.size() == 1)
        n3.no_more = true;
    if (n4.vKeys.size() == 1)
        n4.no_more = true;
}

vector<cv::KeyPoint> Extractor::DistributeQuadTree(
    const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
    const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));
    const float hX = static_cast<float>(maxX - minX) / nIni;
    list<QuadTreeNode> lNodes;
    vector<QuadTreeNode *> vpIniNodes;
    vpIniNodes.resize(nIni);
    for (int i = 0; i < nIni; i++)
    {
        QuadTreeNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    auto lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->vKeys.size() == 1)
        {
            lit->no_more = true;
            lit++;
        }
        else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, QuadTreeNode *>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;
        int prevSize = lNodes.size();
        lit = lNodes.begin();
        int nToExpand = 0;
        vSizeAndPointerToNode.clear();
        while (lit != lNodes.end())
        {
            if (lit->no_more)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                QuadTreeNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        else if (((int)lNodes.size() + nToExpand * 3) > N)
        {

            while (!bFinish)
            {
                prevSize = lNodes.size();
                vector<pair<int, QuadTreeNode *>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();
                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    QuadTreeNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(num_features);
    for (auto lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint *pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++)
        {
            if (vNodeKeys[k].response > maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void Extractor::ComputeKeyPointsQuadTree(vector<vector<KeyPoint>> &allKeypoints)
{
    allKeypoints.resize(num_levels);

    const float W = 30;
    for (int level = 0; level < num_levels; ++level)
    {
        const int minBorderX = edge_thershold - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = image_pyramid_[level].cols - edge_thershold + 3;
        const int maxBorderY = image_pyramid_[level].rows - edge_thershold + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(num_features * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                FAST(image_pyramid_[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                     vKeysCell, init_FAST_thershold, true);

                if (vKeysCell.empty())
                {
                    FAST(image_pyramid_[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, min_FAST_thershold, true);
                }

                if (!vKeysCell.empty())
                {
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        vector<KeyPoint> &keypoints = allKeypoints[level];
        keypoints.reserve(num_features);

        keypoints = DistributeQuadTree(vToDistributeKeys, minBorderX, maxBorderX,
                                       minBorderY, maxBorderY, num_features_per_levels_[level], level);

        const int scaledPatchSize = patch_size * scale_factor_per_levels_[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < num_levels; ++level)
        compute_orientation(image_pyramid_[level], allKeypoints[level], umax);
}

void Extractor::DetectAndCompute(InputArray image_in, InputArray mask_in, vector<KeyPoint> &keypoints, OutputArray descriptors_out)
{
    if (image_in.empty())
        return;

    Mat image = image_in.getMat();
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    vector<vector<KeyPoint>> allKeypoints;
    ComputeKeyPointsQuadTree(allKeypoints);

    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < num_levels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if (nkeypoints == 0)
        descriptors.release();
    else
    {
        descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = descriptors_out.getMat();
    }

    keypoints.clear();
    keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < num_levels; ++level)
    {
        vector<KeyPoint> &keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image
        Mat current_image = image_pyramid_[level].clone();
        GaussianBlur(current_image, current_image, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        orb_->compute(current_image, keypoints, desc);
        offset += nkeypointsLevel;

        // scale keypoint coordinates
        if (level != 0)
        {
            float scale = scale_factor_per_levels_[level];
            for (auto keypoint = keypoints.begin(), keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // add the keypoints to the output
        keypoints.insert(keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void Extractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < num_levels; ++level)
    {
        float scale = inv_scale_factor_per_levels_[level];
        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        Size wholeSize(sz.width + edge_thershold * 2, sz.height + edge_thershold * 2);
        Mat temp(wholeSize, image.type()), masktemp;
        image_pyramid_[level] = temp(Rect(edge_thershold, edge_thershold, sz.width, sz.height));

        // Compute the resized image
        if (level != 0)
        {
            resize(image_pyramid_[level - 1], image_pyramid_[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(image_pyramid_[level], temp, edge_thershold, edge_thershold, edge_thershold, edge_thershold, BORDER_REFLECT_101 + BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, edge_thershold, edge_thershold, edge_thershold, edge_thershold, BORDER_REFLECT_101);
        }
    }
}

} // namespace lvio_fusion
