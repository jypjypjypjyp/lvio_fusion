#ifndef lvio_fusion_NAVSAT_H
#define lvio_fusion_NAVSAT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/sensor.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{
struct NavsatPoint
{
    NavsatPoint() {}
    NavsatPoint(double time, NavsatPoint &last_point, Vector3d A, Vector3d B, Vector3d C, Vector3d position)
        : time(time), position(position), A(A), B(B), C(C)
    {
        heading = position - last_point.position;
    }
    double time;
    Vector3d position;
    Vector3d A, B, C;
    Vector3d heading;
};

class Map;

class NavsatMap
{
public:
    typedef std::map<double, NavsatPoint> NavsatPoints;
    typedef std::shared_ptr<NavsatMap> Ptr;

    NavsatMap(std::shared_ptr<Map> map) : map_(map) {}

    void AddPoint(double time, double x, double y, double z);

    void Transfrom(NavsatPoint &point);

    void Initialize();

    bool initialized = false;
    std::map<double, Vector3d> raw;
    NavsatPoints points;
    SE3d tf;

private:
    bool Check();

    std::weak_ptr<Map> map_;
    NavsatPoint A_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H