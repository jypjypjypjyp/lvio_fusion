#ifndef lvio_fusion_POSE_GRAPH_H
#define lvio_fusion_POSE_GRAPH_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

// [A, B, C]
struct Section
{
    double A; // time of before the first frame
    double B; // time of before the first loop frame
    double C; // the last frame
};

typedef std::map<double, Section> Atlas;

class PoseGraph
{
public:
    typedef std::shared_ptr<PoseGraph> Ptr;

    void AddSubMap(double old_time, double start_time, double end_time);

    std::map<double, SE3d> GetActiveSubMaps(Frames &active_kfs, double &old_time, double start_time);

    Atlas GetSections(double start, double end);

    void BuildProblem(Atlas sections, adapt::Problem &problem);

    void Optimize(adapt::Problem problem);

private:
    void UpdateSections(double time);

    Atlas atlas_;    // loop altas
    Atlas sections_; // sections
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_GRAPH_H
