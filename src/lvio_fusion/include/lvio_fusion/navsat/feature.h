#ifndef lvio_fusion_NAVSAT_FEATURE_H
#define lvio_fusion_NAVSAT_FEATURE_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{
namespace navsat
{

class Feature
{
public:
    typedef std::shared_ptr<Feature> Ptr;

    Feature(double time, Vector3d cov) : time(time), cov(cov) {}

    double time;
    Vector3d cov;
};

} // namespace navsat
} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_FEATURE_H