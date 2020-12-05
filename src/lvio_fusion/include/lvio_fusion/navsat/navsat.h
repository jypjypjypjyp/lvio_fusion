#ifndef lvio_fusion_NAVSAT_H
#define lvio_fusion_NAVSAT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

class Map;

class NavsatMap
{
public:
    typedef std::shared_ptr<NavsatMap> Ptr;

    NavsatMap(std::shared_ptr<Map> map) : map_(map)
    {
        A_ = B_ = C_ = std::make_pair(0, Vector3d(0, 0, 0));
    }

    void AddPoint(double time, double x, double y, double z);

    Vector3d GetPoint(double time);

    bool initialized = false;
    std::map<double, Vector3d> raw;
    SE3d tf;

private:
    bool Check(double time, Vector3d position);

    void Initialize();

    std::weak_ptr<Map> map_;
    std::pair<double, Vector3d> A_, B_, C_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H