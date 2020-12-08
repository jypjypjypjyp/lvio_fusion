#ifndef lvio_fusion_NAVSAT_FEATURE_H
#define lvio_fusion_NAVSAT_FEATURE_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/navsat/navsat.h"

namespace lvio_fusion
{
namespace navsat
{

class Feature
{
public:
    typedef std::shared_ptr<Feature> Ptr;

    Feature(double time, double last, double A, double B, double C, lvio_fusion::NavsatMap *map)
        : time_(time), last_(last), A_(A), B_(B), C_(C), map_(map) {}

    Vector3d Position()
    {
        return map_->GetPoint(time_);
    }

    Vector3d Heading()
    {
        return map_->GetPoint(time_) - map_->GetPoint(last_);
    }

    Vector3d A()
    {
        return map_->GetPoint(A_);
    }

    Vector3d B()
    {
        return map_->GetPoint(B_);
    }

    Vector3d C()
    {
        return map_->GetPoint(C_);
    }

private:
    NavsatMap *map_;
    double time_, last_;
    double A_, B_, C_;
};

} // namespace navsat
} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_FEATURE_H