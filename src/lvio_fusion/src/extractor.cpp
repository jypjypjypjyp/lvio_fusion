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

    num_desired_features_.resize(num_levels);
    float inv_factor = 1.0f / scale_factor;
    float num_desired_features_scale = num_features * (1 - inv_factor) / (1 - (float)pow((double)inv_factor, (double)num_levels));

    int sum = 0;
    for (int i = 0; i < num_levels - 1; i++)
    {
        num_desired_features_[i] = cvRound(num_desired_features_scale);
        sum += num_desired_features_[i];
        num_desired_features_scale *= inv_factor;
    }
    num_desired_features_[num_levels - 1] = std::max(num_features - sum, 0);

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    umax_.resize(half_patch_size + 1);
    int v, v0, vmax = cvFloor(half_patch_size * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(half_patch_size * sqrt(2.f) / 2);
    const double hp2 = half_patch_size * half_patch_size;
    for (v = 0; v <= vmax; ++v)
        umax_[v] = cvRound(sqrt(hp2 - v * v));

    // make sure we are symmetric
    for (v = half_patch_size, v0 = 0; v >= vmin; --v)
    {
        while (umax_[v0] == umax_[v0 + 1])
            ++v0;
        umax_[v] = v0;
        ++v0;
    }
}

inline float Extractor::ICAngle(const Mat &image, Point2f pt)
{
    int m_01 = 0, m_10 = 0;

    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -half_patch_size; u <= half_patch_size; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= half_patch_size; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = umax_[v];
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

inline void Extractor::ComputeOrientation(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
{
    for (auto keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
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
    n1.kps.reserve(kps.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.kps.reserve(kps.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.kps.reserve(kps.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.kps.reserve(kps.size());

    //Associate points to childs
    for (size_t i = 0; i < kps.size(); i++)
    {
        const cv::KeyPoint &kp = kps[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.kps.push_back(kp);
            else
                n3.kps.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.kps.push_back(kp);
        else
            n4.kps.push_back(kp);
    }

    if (n1.kps.size() == 1)
        n1.no_more = true;
    if (n2.kps.size() == 1)
        n2.no_more = true;
    if (n3.kps.size() == 1)
        n3.no_more = true;
    if (n4.kps.size() == 1)
        n4.no_more = true;
}

vector<cv::KeyPoint> Extractor::DistributeQuadTree(
    const vector<cv::KeyPoint> &distribute_kps, const int &min_x,
    const int &max_x, const int &min_y, const int &max_y, const int &num, const int &level)
{
    // compute how many initial nodes
    const int num_init_nodes = round(static_cast<float>(max_x - min_x) / (max_y - min_y));
    const float height = static_cast<float>(max_x - min_x) / num_init_nodes;
    list<QuadTreeNode> list_nodes;
    vector<QuadTreeNode *> init_nodes;
    init_nodes.resize(num_init_nodes);
    for (int i = 0; i < num_init_nodes; i++)
    {
        QuadTreeNode node;
        node.UL = cv::Point2i(height * static_cast<float>(i), 0);
        node.UR = cv::Point2i(height * static_cast<float>(i + 1), 0);
        node.BL = cv::Point2i(node.UL.x, max_y - min_y);
        node.BR = cv::Point2i(node.UR.x, max_y - min_y);
        node.kps.reserve(distribute_kps.size());

        list_nodes.push_back(node);
        init_nodes[i] = &list_nodes.back();
    }

    // associate points to childs
    for (size_t i = 0; i < distribute_kps.size(); i++)
    {
        const cv::KeyPoint &kp = distribute_kps[i];
        init_nodes[kp.pt.x / height]->kps.push_back(kp);
    }

    auto iter = list_nodes.begin();
    while (iter != list_nodes.end())
    {
        if (iter->kps.size() == 1)
        {
            iter->no_more = true;
            iter++;
        }
        else if (iter->kps.empty())
            iter = list_nodes.erase(iter);
        else
            iter++;
    }

    bool finished = false;
    int iteration = 0;
    vector<pair<int, QuadTreeNode *>> size_and_nodes;
    size_and_nodes.reserve(list_nodes.size() * 4);
    while (!finished)
    {
        iteration++;
        int prev_size = list_nodes.size();
        iter = list_nodes.begin();
        int num_expand = 0;
        size_and_nodes.clear();
        while (iter != list_nodes.end())
        {
            if (iter->no_more)
            {
                // If node only contains one point do not subdivide and continue
                iter++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                QuadTreeNode n1, n2, n3, n4;
                iter->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.kps.size() > 0)
                {
                    list_nodes.push_front(n1);
                    if (n1.kps.size() > 1)
                    {
                        num_expand++;
                        size_and_nodes.push_back(make_pair(n1.kps.size(), &list_nodes.front()));
                        list_nodes.front().iter = list_nodes.begin();
                    }
                }
                if (n2.kps.size() > 0)
                {
                    list_nodes.push_front(n2);
                    if (n2.kps.size() > 1)
                    {
                        num_expand++;
                        size_and_nodes.push_back(make_pair(n2.kps.size(), &list_nodes.front()));
                        list_nodes.front().iter = list_nodes.begin();
                    }
                }
                if (n3.kps.size() > 0)
                {
                    list_nodes.push_front(n3);
                    if (n3.kps.size() > 1)
                    {
                        num_expand++;
                        size_and_nodes.push_back(make_pair(n3.kps.size(), &list_nodes.front()));
                        list_nodes.front().iter = list_nodes.begin();
                    }
                }
                if (n4.kps.size() > 0)
                {
                    list_nodes.push_front(n4);
                    if (n4.kps.size() > 1)
                    {
                        num_expand++;
                        size_and_nodes.push_back(make_pair(n4.kps.size(), &list_nodes.front()));
                        list_nodes.front().iter = list_nodes.begin();
                    }
                }

                iter = list_nodes.erase(iter);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)list_nodes.size() >= num || (int)list_nodes.size() == prev_size)
        {
            finished = true;
        }
        else if (((int)list_nodes.size() + num_expand * 3) > num)
        {

            while (!finished)
            {
                prev_size = list_nodes.size();
                vector<pair<int, QuadTreeNode *>> prev_size_and_node = size_and_nodes;
                size_and_nodes.clear();
                sort(prev_size_and_node.begin(), prev_size_and_node.end());
                for (int j = prev_size_and_node.size() - 1; j >= 0; j--)
                {
                    QuadTreeNode n1, n2, n3, n4;
                    prev_size_and_node[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.kps.size() > 0)
                    {
                        list_nodes.push_front(n1);
                        if (n1.kps.size() > 1)
                        {
                            size_and_nodes.push_back(make_pair(n1.kps.size(), &list_nodes.front()));
                            list_nodes.front().iter = list_nodes.begin();
                        }
                    }
                    if (n2.kps.size() > 0)
                    {
                        list_nodes.push_front(n2);
                        if (n2.kps.size() > 1)
                        {
                            size_and_nodes.push_back(make_pair(n2.kps.size(), &list_nodes.front()));
                            list_nodes.front().iter = list_nodes.begin();
                        }
                    }
                    if (n3.kps.size() > 0)
                    {
                        list_nodes.push_front(n3);
                        if (n3.kps.size() > 1)
                        {
                            size_and_nodes.push_back(make_pair(n3.kps.size(), &list_nodes.front()));
                            list_nodes.front().iter = list_nodes.begin();
                        }
                    }
                    if (n4.kps.size() > 0)
                    {
                        list_nodes.push_front(n4);
                        if (n4.kps.size() > 1)
                        {
                            size_and_nodes.push_back(make_pair(n4.kps.size(), &list_nodes.front()));
                            list_nodes.front().iter = list_nodes.begin();
                        }
                    }

                    list_nodes.erase(prev_size_and_node[j].second->iter);

                    if ((int)list_nodes.size() >= num)
                        break;
                }

                if ((int)list_nodes.size() >= num || (int)list_nodes.size() == prev_size)
                    finished = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> result;
    result.reserve(num_features);
    for (auto iter = list_nodes.begin(); iter != list_nodes.end(); iter++)
    {
        vector<cv::KeyPoint> &kps = iter->kps;
        cv::KeyPoint *kp = &kps[0];
        float max_response = kp->response;
        for (int i = 1; i < kps.size(); i++)
        {
            if (kps[i].response > max_response)
            {
                kp = &kps[i];
                max_response = kps[i].response;
            }
        }
        result.push_back(*kp);
    }

    return result;
}

void Extractor::ComputeKeyPointsQuadTree(vector<vector<KeyPoint>> &all_kps)
{
    all_kps.resize(num_levels);

    const float W = 30;
    for (int level = 0; level < num_levels; level++)
    {
        const int min_border_x = edge_thershold - 3;
        const int min_border_Y = min_border_x;
        const int max_border_X = image_pyramid_[level].cols - edge_thershold + 3;
        const int max_border_Y = image_pyramid_[level].rows - edge_thershold + 3;

        vector<cv::KeyPoint> distribute_kps;
        distribute_kps.reserve(num_features * 10);

        const float width = (max_border_X - min_border_x);
        const float height = (max_border_Y - min_border_Y);
        const int cols = width / W;
        const int rows = height / W;
        const int cell_width = ceil(width / cols);
        const int cell_height = ceil(height / rows);

        for (int i = 0; i < rows; i++)
        {
            const float init_y = min_border_Y + i * cell_height;
            float max_y = init_y + cell_height + 6;

            if (init_y >= max_border_Y - 3)
                continue;
            if (max_y > max_border_Y)
                max_y = max_border_Y;

            for (int j = 0; j < cols; j++)
            {
                const float init_x = min_border_x + j * cell_width;
                float max_x = init_x + cell_width + 6;
                if (init_x >= max_border_X - 6)
                    continue;
                if (max_x > max_border_X)
                    max_x = max_border_X;

                vector<cv::KeyPoint> cell_kps;
                FAST(image_pyramid_[level].rowRange(init_y, max_y).colRange(init_x, max_x),
                     cell_kps, init_FAST_thershold, true);

                if (cell_kps.empty())
                {
                    FAST(image_pyramid_[level].rowRange(init_y, max_y).colRange(init_x, max_x),
                         cell_kps, min_FAST_thershold, true);
                }

                if (!cell_kps.empty())
                {
                    for (auto vit = cell_kps.begin(); vit != cell_kps.end(); vit++)
                    {
                        (*vit).pt.x += j * cell_width;
                        (*vit).pt.y += i * cell_height;
                        distribute_kps.push_back(*vit);
                    }
                }
            }
        }

        vector<KeyPoint> &level_kps = all_kps[level];
        level_kps.reserve(num_features);

        level_kps = DistributeQuadTree(distribute_kps, min_border_x, max_border_X,
                                       min_border_Y, max_border_Y, num_desired_features_[level], level);

        const int scaled_patch_size = patch_size * scale_factor_per_levels_[level];

        // Add border to coordinates and scale information
        const int nkps = level_kps.size();
        for (int i = 0; i < nkps; i++)
        {
            level_kps[i].pt.x += min_border_x;
            level_kps[i].pt.y += min_border_Y;
            level_kps[i].octave = level;
            level_kps[i].size = scaled_patch_size;
        }
    }

    // compute orientations
    for (int level = 0; level < num_levels; level++)
        ComputeOrientation(image_pyramid_[level], all_kps[level]);
}

void Extractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < num_levels; level++)
    {
        float scale = inv_scale_factor_per_levels_[level];
        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        Size whole_size(sz.width + edge_thershold * 2, sz.height + edge_thershold * 2);
        Mat temp(whole_size, image.type());
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

void Extractor::Detect(Mat image, vector<vector<KeyPoint>> &keypoints)
{
    keypoints.clear();
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    ComputeKeyPointsQuadTree(keypoints);

    for (int level = 1; level < num_levels; level++)
    {
        vector<KeyPoint> &kps_level = keypoints[level];
        if (kps_level.empty())
            continue;

        // scale keypoint coordinates
        float scale = scale_factor_per_levels_[level];
        for (auto &kp : kps_level)
        {
            kp.pt *= scale;
        }
    }
}

Mat Extractor::Compute(vector<vector<KeyPoint>> &keypoints)
{
    static Ptr<ORB> orb = ORB::create();
    int num_kps = 0;
    for (auto &kps_level : keypoints)
        num_kps += kps_level.size();

    Mat descriptors = Mat::zeros(num_kps, 32, CV_8U);
    int offset = 0;
    for (int level = 0; level < num_levels; level++)
    {
        vector<KeyPoint> &kps_level = keypoints[level];
        int size = kps_level.size();
        if (size == 0)
            continue;

        // preprocess the resized image
        Mat current_image = image_pyramid_[level].clone();
        GaussianBlur(current_image, current_image, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + size);
        orb->compute(current_image, kps_level, desc);
        offset += size;
    }
    return descriptors;
}

} // namespace lvio_fusion
