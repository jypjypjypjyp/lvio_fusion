#ifndef lvio_fusion_NAVSAT_H
#define lvio_fusion_NAVSAT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

class Navsat : public Sensor
{
public:
    typedef std::shared_ptr<Navsat> Ptr;

    static int Create()
    {
        devices_.push_back(Navsat::Ptr(new Navsat));
        return devices_.size() - 1;
    }

    static int Num()
    {
        return devices_.size();
    }

    static Navsat::Ptr Get(int id = 0)
    {
        return devices_[id];
    }

    void SetPoseGraph(PoseGraph::Ptr pose_graph) { pose_graph_ = pose_graph; }

    void AddPoint(double time, double x, double y, double z);

    Vector3d GetPoint(double time);

    Atlas Optimize(double time);

    bool initialized = false;
    std::map<double, Vector3d> raw;

private:
    Navsat() : Sensor(SE3d())
    {
        A_ = B_ = C_ = std::make_pair(0, Vector3d(0, 0, 0));
    }
    Navsat(const Navsat &);
    Navsat &operator=(const Navsat &);

    bool UpdateLevel(double time, Vector3d position);

    void Initialize();

    PoseGraph::Ptr pose_graph_;
    std::pair<double, Vector3d> A_, B_, C_; // there points on the level
    static std::vector<Navsat::Ptr> devices_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H