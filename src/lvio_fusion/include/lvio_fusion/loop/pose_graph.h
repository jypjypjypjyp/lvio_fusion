#ifndef lvio_fusion_POSE_GRAPH_H
#define lvio_fusion_POSE_GRAPH_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

// [old_time, start_time, end_time]]
struct Submap
{
    double start_time; // time of before the first loop frame
    double end_time;   // the last frame
    double old_time;   // time of before the first frame
};

typedef std::map<double, Submap> Atlas;

class PoseGraph
{
public:
    typedef std::shared_ptr<PoseGraph> Ptr;

    void AddSubMap(double old_time, double start_time, double end_time);

    std::map<double, SE3d> GetActiveSubMaps(Frames &active_kfs, double &old_time, double start_time);

    Atlas GetSections(Frames active_kfs);

private:

    void UpdateSections(double time);

    Atlas altas_;  // submaps for loop
    Atlas sections_; // submaps for pose graph optimizing
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_GRAPH_H
