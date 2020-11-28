#ifndef lvio_fusion_MAPPING_H
#define lvio_fusion_MAPPING_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.hpp"

namespace lvio_fusion
{

class Mapping
{
public:
    typedef std::shared_ptr<Mapping> Ptr;

    Mapping() {}

    void SetLidar(Lidar::Ptr lidar) { lidar_ = lidar; }

    void SetCamera(Camera::Ptr camera) { camera_ = camera; }

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetFeatureAssociation(FeatureAssociation::Ptr association) { association_ = association; }

    void Optimize(Frames &active_kfs);

    void BuildOldMapFrame(Frames old_frames, Frame::Ptr map_frame);

    void MergeScan(const PointICloud &in, SE3d from_pose, PointICloud &out);

    void AddToWorld(Frame::Ptr frame);

    PointRGBCloud GetGlobalMap();

    std::map<double, PointRGBCloud> pointclouds_color;
    std::map<double, PointICloud> pointclouds_full;

private:
    void BuildMapFrame(Frame::Ptr frame, Frame::Ptr map_frame);

    void Color(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out);

    Map::Ptr map_;
    FeatureAssociation::Ptr association_;

    Lidar::Ptr lidar_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
