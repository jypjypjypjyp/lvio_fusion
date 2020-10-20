#ifndef lvio_fusion_ATLAS_H
#define lvio_fusion_LOOP_CONSTRAINT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

namespace loop
{

// (start_time, end_time]
struct SubMap
{
    double start_time;
    double end_time;
    double old_time;
};

class Atlas
{
public:
    void AddSubMap(double old_time, double start_time, double end_time);

    std::map<double, SE3d> GetActiveSubMaps(Frames& active_kfs, double& old_time, double start_time, double end_time);

private:
    std::map<double, SubMap> submaps_;
};

} // namespace loop
} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_CONSTRAINT_H
