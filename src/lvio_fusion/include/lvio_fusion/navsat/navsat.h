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

    static int Create(double accuracy, bool navsat_v)
    {
        devices_.push_back(Navsat::Ptr(new Navsat(accuracy, navsat_v)));
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

    void AddPoint(double time, double x, double y, double z, Vector3d cov);

    Vector3d GetFixPoint(Frame::Ptr frame);
    Vector3d GetRawPoint(double time);
    Vector3d GetPoint(double time);
    Vector3d GetAroundPoint(double time);

    void Optimize(const Section &section);
    void QuickFix(double start, double end);

    bool initialized = false;
    std::map<double, Vector3d> raw;
    Vector3d fix = Vector3d::Zero();
    bool navsat_v;

private:
    Navsat(double accuracy, bool navsat_v) : accuracy_(accuracy), navsat_v(navsat_v), min_distance_fix_(20), Sensor(SE3d()) {}
    Navsat(const Navsat &);
    Navsat &operator=(const Navsat &);

    bool EstimatePose(double time, SE3d &pose);

    void Initialize();

    // mode: y p r x y z;
    void OptimizeRX(Frame::Ptr frame, double end, double forward, unsigned char mode);
    void OptimizeAB(Frame::Ptr A, Frame::Ptr B, SE3d relative_B);

    static std::vector<Navsat::Ptr> devices_;
    double accuracy_;
    double min_distance_fix_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H