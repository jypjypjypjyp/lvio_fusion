#ifndef lvio_fusion_ATLAS_H
#define lvio_fusion_ATLAS_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

namespace loop
{

// (old_time, (start_time, end_time]]
struct SubMap
{
    double start_time;      // time of before the first loop frame
    double end_time;        // the last frame
    double old_time;        // time of before the first frame
};

class Atlas
{
public:
    void AddSubMap(double old_time, double start_time, double end_time);

    std::map<double, SE3d> GetActiveSubMaps(Frames &active_kfs, double &old_time, double start_time);

private:
    std::map<double, SubMap> submaps_;
};

} // namespace loop
} // namespace lvio_fusion

#endif // lvio_fusion_ATLAS_H
