#ifndef lvio_fusion_NAVSAT_H
#define lvio_fusion_NAVSAT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

class NavsatMap : public Sensor
{
public:
    typedef std::shared_ptr<NavsatMap> Ptr;

    NavsatMap() : Sensor(SE3d())
    {
        A_ = B_ = C_ = std::make_pair(0, Vector3d(0, 0, 0));
    }

    void SetPoseGraph(PoseGraph::Ptr pose_graph)
    {
        pose_graph_ = pose_graph;
    }

    void AddPoint(double time, double x, double y, double z);

    Vector3d GetPoint(double time);

    void Optimize(Frames active_kfs);

    bool initialized = false;
    std::map<double, Vector3d> raw;

private:
    bool UpdateLevel(double time, Vector3d position);

    void Initialize();

    PoseGraph::Ptr pose_graph_;
    std::pair<double, Vector3d> A_, B_, C_; // there points on the level
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H