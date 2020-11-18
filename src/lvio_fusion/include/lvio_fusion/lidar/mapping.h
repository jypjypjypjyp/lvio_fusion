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

    PointRGBCloud GetGlobalMap();

private:
    enum class ProblemType
    {
        Ground,
        Segmented,
    };

    void MappingLoop();

    void BuildMapFrame(Frame::Ptr frame, Frame::Ptr map_frame);

    void BuildProblem(Frame::Ptr frame, Frame::Ptr map_frame, double *para, ceres::Problem &problem, Mapping::ProblemType type);

    void AddToWorld(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out);

    void BuildGlobalMap(Frames &active_kfs);

    Map::Ptr map_;
    FeatureAssociation::Ptr association_;

    std::map<double, PointRGBCloud> pointclouds_;

    Lidar::Ptr lidar_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
