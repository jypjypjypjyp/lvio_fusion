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
    NavsatPoint(double time, double x, double y, double z)
        : time(time), position(Vector3d(x, y, z)) {}
    double time;
    Vector3d position;
};

class Map;

class NavsatMap
{
public:
    typedef std::map<double, NavsatPoint> NavsatPoints;
    typedef std::shared_ptr<NavsatMap> Ptr;

    NavsatMap(std::shared_ptr<Map> map) : map_(map) {}

    void AddPoint(NavsatPoint point)
    {
        navsat_points[point.time] = point;
    }

    void Transfrom(NavsatPoint &point)
    {
        point.position = tf * point.position;
    }

    void Initialize();

    bool initialized = false;
    int num_frames_init = 40;
    NavsatPoints navsat_points;
    SE3d tf;

private:
    std::weak_ptr<Map> map_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H