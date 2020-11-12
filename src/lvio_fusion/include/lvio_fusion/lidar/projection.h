#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class ImageProjection
{
public:
    ImageProjection(int num_scans, double cycle_time, double min_range, double max_range, double deskew) : num_scans_(num_scans), cycle_time_(cycle_time), min_range_(min_range), max_range_(max_range), deskew_(deskew) {}

    void SetLidar(Lidar::Ptr lidar) { lidar_ = lidar; }

    PointICloud::Ptr laserCloudIn;
    PointICloud::Ptr fullCloud;     // projected velodyne raw cloud, but saved in the form of 1-D matrix
    PointICloud::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    PointICloud::Ptr groundCloud;
    PointICloud::Ptr segmentedCloud;
    PointICloud::Ptr segmentedCloudPure;
    PointICloud::Ptr outlierCloud;

    PointI nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat;  // range matrix for range image
    cv::Mat labelMat;  // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    std::vector<std::pair<int8_t, int8_t>> neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

    void Preprocess(PointICloud &points, Frame::Ptr frame);

private:
    void AddScan(double time, Point3Cloud::Ptr new_scan);

    bool TimeAlign(double time, PointICloud &out);

    void allocateMemory();

    void resetParameters();

    void findStartEndAngle();

    void projectPointCloud();

    void groundRemoval();

    void cloudSegmentation();

    void labelComponents();

    Map::Ptr map_;
    std::map<double, Point3Cloud::Ptr> raw_point_clouds_;
    Lidar::Ptr lidar_;

    // params
    const int num_scans_;
    const double cycle_time_;
    const double min_range_;
    const double max_range_;
    const bool deskew_;

    const int Horizon_SCAN = 1800;
    const float ang_res_x = 360.0/float(Horizon_SCAN);
    const float ang_res_y = 41.33/float(num_scans_-1);
    const float ang_bottom = 30.67;
    const int groundScanInd = 20;
};

} // namespace lvio_fusion