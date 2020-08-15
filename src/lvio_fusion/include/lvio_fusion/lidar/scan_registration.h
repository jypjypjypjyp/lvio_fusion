#ifndef lvio_fusion_REGISTRATION_H
#define lvio_fusion_REGISTRATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class Frontend;

class ScanRegistration
{
public:
    typedef std::shared_ptr<ScanRegistration> Ptr;

    ScanRegistration::ScanRegistration(int num_scan, double cycle_time, double minimum_range, double deskew) : num_scans_(num_scan), cycle_time_(cycle_time), minimum_range_(minimum_range), deskew_(deskew) {}

    void SetLidar(Lidar::Ptr lidar)
    {
        lidar_ = lidar;
    }

    void SetMap(Map::Ptr map)
    {
        map_ = map;
    }

    void AddScan(double time, Point3Cloud::Ptr new_scan);

private:
    void UndistortPoint(PointI &p, Frame::Ptr frame);

    void Deskew(PointICloud &pc, Frame::Ptr frame);

    bool TimeAlign(double time, PointICloud &out);

    void Preprocess(PointICloud &pc, Frame::Ptr frame);

    Map::Ptr map_;
    std::map<double, Point3Cloud::Ptr> raw_point_clouds_;
    double header = 0; // header of the frames' time which already has a point cloud
    Lidar::Ptr lidar_;

    // params
    const int num_scans_;
    const double cycle_time_;
    const double minimum_range_;
    const bool deskew_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_REGISTRATION_H
