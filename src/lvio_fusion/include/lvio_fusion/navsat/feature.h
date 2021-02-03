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

    Feature(double time) : time(time) {}

    double time;
};

} // namespace navsat
} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_FEATURE_H