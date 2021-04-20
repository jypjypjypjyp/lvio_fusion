#include "lvio_fusion/lidar/projection.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/voxel_grid.h>

#define OUTLIER_LABEL 999999

namespace lvio_fusion
{

void ImageProjection::Clear()
{
    range_mat = cv::Mat(num_scans_, horizon_scan_, CV_32F, cv::Scalar::all(FLT_MAX));
    ground_mat = cv::Mat(num_scans_, horizon_scan_, CV_8S, cv::Scalar::all(0));
    label_mat = cv::Mat(num_scans_, horizon_scan_, CV_32S, cv::Scalar::all(0));
    label_count = 1;
    PointI nan; // fill in fullCloud at each iteration
    nan.x = std::numeric_limits<float>::quiet_NaN();
    nan.y = std::numeric_limits<float>::quiet_NaN();
    nan.z = std::numeric_limits<float>::quiet_NaN();
    nan.intensity = -1;
    std::fill(points_full.points.begin(), points_full.points.end(), nan);
}

SegmentedInfo ImageProjection::Process(PointICloud &points, PointICloud &points_segmented)
{
    SegmentedInfo segmented_info(num_scans_, horizon_scan_);

    FindStartEndAngle(segmented_info, points);

    ProjectPointCloud(segmented_info, points);

    RemoveGround(segmented_info);

    Segment(segmented_info, points_segmented);
    
    if(gridmap_)
        Compute2DScanMsg();//NAVI

    Clear();
    return segmented_info;
}

void ImageProjection::FindStartEndAngle(SegmentedInfo &segmented_info, PointICloud &points)
{
    // start and end orientation of this cloud
    segmented_info.start_orientation = -atan2(points[0].y, points[0].x);
    segmented_info.end_orientation = -atan2(points[points.points.size() - 1].y,
                                            points[points.points.size() - 1].x) +
                                     2 * M_PI;
    if (segmented_info.end_orientation - segmented_info.start_orientation > 3 * M_PI)
    {
        segmented_info.end_orientation -= 2 * M_PI;
    }
    else if (segmented_info.end_orientation - segmented_info.start_orientation < M_PI)
        segmented_info.end_orientation += 2 * M_PI;
    segmented_info.orientation_diff = segmented_info.end_orientation - segmented_info.start_orientation;
}

void ImageProjection::ProjectPointCloud(SegmentedInfo &segmented_info, PointICloud &points)
{
    // range image projection
    float vertical_angle, horizon_angle, range;
    int row_ind, column_ind, index, size;
    PointI point;

    size = points.points.size();

    for (int i = 0; i < size; ++i)
    {
        point.x = points[i].x;
        point.y = points[i].y;
        point.z = points[i].z;
        // find the row and column index in the image for this point

        vertical_angle = atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        row_ind = (vertical_angle + ang_bottom_) / ang_res_y_;

        if (row_ind < 0 || row_ind >= num_scans_)
            continue;

        horizon_angle = atan2(point.x, point.y) * 180 / M_PI;

        column_ind = -round((horizon_angle - 90.0) / ang_res_x_) + horizon_scan_ / 2;
        if (column_ind >= horizon_scan_)
            column_ind -= horizon_scan_;

        if (column_ind < 0 || column_ind >= horizon_scan_)
            continue;

        range = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

        range_mat.at<float>(row_ind, column_ind) = range;

        point.intensity = (float)row_ind + (float)column_ind / 10000.0;

        index = column_ind + row_ind * horizon_scan_;
        points_full[index] = point;
    }
}

void ImageProjection::RemoveGround(SegmentedInfo &segmented_info)
{
    int lower_ind, upper_ind;
    float dx, dy, dz, angle;
    // groundMat
    // -1, no valid info to check if ground of not
    //  0, initial value, after validation, means not ground
    //  1, ground
    for (int j = 0; j < horizon_scan_; ++j)
    {
        for (int i = 0; i < ground_rows_; ++i)
        {

            lower_ind = j + (i)*horizon_scan_;
            upper_ind = j + (i + 1) * horizon_scan_;

            if (points_full[lower_ind].intensity == -1 ||
                points_full[upper_ind].intensity == -1)
            {
                // no info to check, invalid points
                ground_mat.at<int8_t>(i, j) = -1;
                continue;
            }

            dx = points_full[upper_ind].x - points_full[lower_ind].x;
            dy = points_full[upper_ind].y - points_full[lower_ind].y;
            dz = points_full[upper_ind].z - points_full[lower_ind].z;

            angle = atan2(dz, sqrt(dx * dx + dy * dy)) * 180 / M_PI;

            //NOTE: mount angle
            if (abs(angle) <= 10)
            {
                ground_mat.at<int8_t>(i, j) = 1;
                ground_mat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }
    // extract ground cloud (groundMat == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~num_scans_-1, need rangeMat for mark label matrix for the 16th scan
    for (int i = 0; i < num_scans_; ++i)
    {
        for (int j = 0; j < horizon_scan_; ++j)
        {
            if (ground_mat.at<int8_t>(i, j) == 1 || range_mat.at<float>(i, j) == FLT_MAX)
            {
                label_mat.at<int>(i, j) = -1;
            }
        }
    }
}

void ImageProjection::Segment(SegmentedInfo &segmented_info, PointICloud &points_segmented)
{
    // segmentation process
    for (int i = 0; i < num_scans_; ++i)
        for (int j = 0; j < horizon_scan_; ++j)
            if (label_mat.at<int>(i, j) == 0)
                LabelComponents(i, j);

    int num_segmented = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < num_scans_; ++i)
    {

        segmented_info.start_ring_index[i] = num_segmented - 1 + 5;

        for (int j = 0; j < horizon_scan_; ++j)
        {
            if (label_mat.at<int>(i, j) > 0 || ground_mat.at<int8_t>(i, j) == 1)
            {
                // outliers that will not be used for optimization (always continue)
                if (label_mat.at<int>(i, j) == OUTLIER_LABEL)
                {
                    continue;
                }
                // majority of ground points are skipped
                // if (groundMat.at<int8_t>(i, j) == 1)
                // {
                //     if (j % 5 != 0 && j > 5 && j < horizon_scan_ - 5)
                //         continue;
                // }
                // mark ground points so they will not be considered as edge features later
                segmented_info.ground_flag[num_segmented] = (ground_mat.at<int8_t>(i, j) == 1);
                // mark the points' column index for marking occlusion later
                segmented_info.col_ind[num_segmented] = j;
                // save range info
                segmented_info.range[num_segmented] = range_mat.at<float>(i, j);
                // save seg cloud
                points_segmented.push_back(points_full[j + i * horizon_scan_]);
                // size of seg cloud
                ++num_segmented;
            }
        }

        segmented_info.end_ring_index[i] = num_segmented - 1 - 5;
    }
}

void ImageProjection::LabelComponents(int row, int col)
{
    static uint16_t *all_pushed_ind_X = new uint16_t[num_scans_ * horizon_scan_]; // array for tracking points of a segmented object
    static uint16_t *all_pushed_ind_y = new uint16_t[num_scans_ * horizon_scan_];
    static uint16_t *queue_ind_x = new uint16_t[num_scans_ * horizon_scan_]; // array for breadth-first search process of segmentation, for speed
    static uint16_t *queue_ind_y = new uint16_t[num_scans_ * horizon_scan_];
    static std::vector<std::pair<int8_t, int8_t>> neighbors; // neighbor iterator for segmentaiton process
    if (neighbors.empty())
    {
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1;
        neighbor.second = 0;
        neighbors.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = 1;
        neighbors.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = -1;
        neighbors.push_back(neighbor);
        neighbor.first = 1;
        neighbor.second = 0;
        neighbors.push_back(neighbor);
    }

    // use std::queue std::vector std::deque will slow the program down greatly
    float d1, d2, alpha, angle;
    int from_ind_x, from_ind_y, this_ind_x, this_ind_y;
    std::vector<bool> line_count_flag(num_scans_, false);

    queue_ind_x[0] = row;
    queue_ind_y[0] = col;
    int queue_size = 1;
    int queue_start_ind = 0;
    int queue_end_ind = 1;

    all_pushed_ind_X[0] = row;
    all_pushed_ind_y[0] = col;
    int all_pushed_ind_size = 1;

    while (queue_size > 0)
    {
        // Pop point
        from_ind_x = queue_ind_x[queue_start_ind];
        from_ind_y = queue_ind_y[queue_start_ind];
        --queue_size;
        ++queue_start_ind;
        // Mark popped point
        label_mat.at<int>(from_ind_x, from_ind_y) = label_count;
        // Loop through all the neighboring grids of popped grid
        for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
        {
            // new index
            this_ind_x = from_ind_x + (*iter).first;
            this_ind_y = from_ind_y + (*iter).second;
            // index should be within the boundary
            if (this_ind_x < 0 || this_ind_x >= num_scans_)
                continue;
            // at range image margin (left or right side)
            if (this_ind_y < 0)
                this_ind_y = horizon_scan_ - 1;
            if (this_ind_y >= horizon_scan_)
                this_ind_y = 0;
            // prevent infinite loop (caused by put already examined point back)
            if (label_mat.at<int>(this_ind_x, this_ind_y) != 0)
                continue;

            d1 = std::max(range_mat.at<float>(from_ind_x, from_ind_y),
                          range_mat.at<float>(this_ind_x, this_ind_y));
            d2 = std::min(range_mat.at<float>(from_ind_x, from_ind_y),
                          range_mat.at<float>(this_ind_x, this_ind_y));

            if ((*iter).first == 0)
                alpha = segment_alpha_x_;
            else
                alpha = segment_alpha_y_;

            angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

            if (angle > theta)
            {

                queue_ind_x[queue_end_ind] = this_ind_x;
                queue_ind_y[queue_end_ind] = this_ind_y;
                ++queue_size;
                ++queue_end_ind;

                label_mat.at<int>(this_ind_x, this_ind_y) = label_count;
                line_count_flag[this_ind_x] = true;

                all_pushed_ind_X[all_pushed_ind_size] = this_ind_x;
                all_pushed_ind_y[all_pushed_ind_size] = this_ind_y;
                ++all_pushed_ind_size;
            }
        }
    }

    // check if this segment is valid
    bool feasible_segment = false;
    if (all_pushed_ind_size >= 30)
        feasible_segment = true;
    else if (all_pushed_ind_size >= num_segment_valid_points_)
    {
        int count = 0;
        for (int i = 0; i < num_scans_; ++i)
            if (line_count_flag[i] == true)
                ++count;
        if (count >= num_segment_valid_lines_)
            feasible_segment = true;
    }
    // segment is valid, mark these points
    if (feasible_segment == true)
    {
        ++label_count;
    }
    else
    { // segment is invalid, mark these points
        for (int i = 0; i < all_pushed_ind_size; ++i)
        {
            label_mat.at<int>(all_pushed_ind_X[i], all_pushed_ind_y[i]) = OUTLIER_LABEL;
        }
    }
}
//NAVI
void ImageProjection::Compute2DScanMsg()
{
    PointICloud scan_msg;
    for (int j = 0; j < horizon_scan_; ++j) {
        float min_range = 1000;
        int id_min = 0;
        for (int i = 0; i < num_scans_; ++i) {
            int Ind = j + (i)*horizon_scan_;
            float Z = points_full[Ind].z;
            if ((ground_mat.at<int8_t>(i, j) != 1) &&
                (Z > 0.4) && (Z<1.2) &&
                (range_mat.at<float>(i, j)<40)) {                     // 地面上点云忽略, 过高过矮的点忽略， 过远的点忽略
                if(range_mat.at<float>(i, j) < min_range) {           // 计算最小距离
                    min_range = range_mat.at<float>(i, j);
                    id_min = Ind;
                }
            }
        }
        if (min_range<1000) {
            scan_msg.push_back(points_full[id_min]);
        }
    }
    gridmap_->AddScan(scan_msg);
    LOG(INFO)<<"ADDSCAN TO GRIDMAP.  SIZE:  "<<scan_msg.size();
}

} // namespace lvio_fusion