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

    Feature(double time, double last, double A, double B, double C)
        : time(time), last(last), A(A), B(B), C(C) {}

    double time, last;
    double A, B, C;
};

} // namespace navsat
} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_FEATURE_H