#ifndef lvio_fusion_MAPPING_H
#define lvio_fusion_MAPPING_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/association.h"

namespace lvio_fusion
{

class Mapping
{
public:
    typedef std::shared_ptr<Mapping> Ptr;

    Mapping() {}

    void SetFeatureAssociation(FeatureAssociation::Ptr association) { association_ = association; }

    void Optimize(Frames &active_kfs);

    void BuildOldMapFrame(Frame::Ptr old_frame, Frame::Ptr map_frame);

    void MergeScan(const PointICloud &in, SE3d from_pose, PointICloud &out);

    void BuildMapFrame(Frame::Ptr frame, Frame::Ptr map_frame);

    void ToWorld(Frame::Ptr frame);

    int Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame, SE3d &relative_o_c);

    PointRGBCloud GetGlobalMap();

    std::map<double, PointRGBCloud> pointclouds_color;
    std::map<double, PointICloud> pointclouds_surf;
    std::map<double, PointICloud> pointclouds_ground;

private:
    void Color(const PointICloud &points_ground, const PointICloud &points_surf, Frame::Ptr frame, PointRGBCloud &out);

    FeatureAssociation::Ptr association_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
