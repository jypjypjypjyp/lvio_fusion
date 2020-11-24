#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class SegmentedInfo
{
public:
    SegmentedInfo(double num_scans, double horizon_scan)
    {
        start_ring_index.assign(num_scans, 0);
        end_ring_index.assign(num_scans, 0);
        ground_flag.assign(num_scans * horizon_scan, false);
        col_ind.assign(num_scans * horizon_scan, 0);
        range.assign(num_scans * horizon_scan, 0);
    }

    std::vector<int> start_ring_index;
    std::vector<int> end_ring_index;

    float start_orientation;
    float end_orientation;
    float orientation_diff;

    std::vector<bool> ground_flag;     // true - ground point, false - other points
    std::vector<unsigned int> col_ind; // point column index in range image
    std::vector<float> range;         // point range
};

class ImageProjection
{
public:
    typedef std::shared_ptr<ImageProjection> Ptr;

    ImageProjection(int num_scans, int horizon_scan, double ang_res_y, double ang_bottom, int ground_rows)
        : num_scans_(num_scans), horizon_scan_(horizon_scan),
          ang_res_x_(360.0 / float(horizon_scan)), ang_res_y_(ang_res_y), ang_bottom_(ang_bottom),
          ground_rows_(ground_rows),
          segment_alpha_x_(ang_res_x_ / 180.0 * M_PI), segment_alpha_y_(ang_res_y_ / 180.0 * M_PI)
    {
        points_full.points.resize(num_scans_ * horizon_scan);
        Clear();
    }

    SegmentedInfo Process(PointICloud &points, PointICloud &points_segmented);

private:
    void FindStartEndAngle(SegmentedInfo &segmented_info, PointICloud& points);

    void ProjectPointCloud(SegmentedInfo &segmented_info, PointICloud& points);

    void RemoveGround(SegmentedInfo &segmented_info);

    void Segment(SegmentedInfo &segmented_info, PointICloud &points_segmented);

    void LabelComponents(int row, int col);

    void Clear();

    PointICloud points_full; // projected velodyne raw cloud, but saved in the form of 1-D matrix

    cv::Mat range_mat;  // range matrix for range image
    cv::Mat label_mat;  // label matrix for segmentaiton marking
    cv::Mat ground_mat; // ground matrix for ground cloud marking
    int label_count;

    // params
    const int num_scans_;
    const int horizon_scan_;
    const float ang_res_x_;
    const float ang_res_y_;
    const float ang_bottom_;
    const int ground_rows_;

    const float theta = 60.0 / 180.0 * M_PI; // decrese this value may improve accuracy
    const int num_segment_valid_points_ = 5;
    const int num_segment_valid_lines_ = 3;
    const float segment_alpha_x_;
    const float segment_alpha_y_;
};

} // namespace lvio_fusion