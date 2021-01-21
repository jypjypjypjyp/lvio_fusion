#ifndef lvio_fusion_REGISTRATION_H
#define lvio_fusion_REGISTRATION_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/lidar/projection.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

class Frontend;

class FeatureAssociation
{
public:
    typedef std::shared_ptr<FeatureAssociation> Ptr;

    FeatureAssociation(int num_scans, int horizon_scan, double ang_res_y, double ang_bottom, int ground_rows, double cycle_time, double min_range, double max_range, double deskew)
        : num_scans_(num_scans), cycle_time_(cycle_time), min_range_(min_range), max_range_(max_range), deskew_(deskew)
    {
        curvatures = new float[num_scans * horizon_scan];
        projection_ = ImageProjection::Ptr(new ImageProjection(num_scans, horizon_scan, ang_res_y, ang_bottom, ground_rows));
    }

    void AddScan(double time, Point3Cloud::Ptr new_scan);

    void ScanToMapWithGround(Frame::Ptr frame, Frame::Ptr map_frame, double *para, adapt::Problem &problem);

    void ScanToMapWithSegmented(Frame::Ptr frame, Frame::Ptr map_frame, double *para, adapt::Problem &problem);
    void SegmentGround(PointICloud &points_ground);

private:
    void UndistortPoint(PointI &point, Frame::Ptr frame);
    void UndistortPointCloud(PointICloud &points, Frame::Ptr frame);

    bool AlignScan(double time, PointICloud &out);

    void Process(PointICloud &points, Frame::Ptr frame);

    void Preprocess(PointICloud &points);

    void Extract(PointICloud &points_segmented, SegmentedInfo &segemented_info, Frame::Ptr frame);

    void AdjustDistortion(PointICloud &points_segmented, SegmentedInfo &segemented_info);

    void CalculateSmoothness(PointICloud &points_segmented, SegmentedInfo &segemented_info);

    void ExtractFeatures(PointICloud &points_segmented, SegmentedInfo &segemented_info, Frame::Ptr frame);

    void Sensor2Robot(PointICloud &in, PointICloud &out);

    ImageProjection::Ptr projection_;
    std::map<double, Point3Cloud::Ptr> raw_point_clouds_;
    float *curvatures;

    // params
    const double num_scans_;
    const double cycle_time_;
    const double min_range_;
    const double max_range_;
    const bool deskew_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_REGISTRATION_H
