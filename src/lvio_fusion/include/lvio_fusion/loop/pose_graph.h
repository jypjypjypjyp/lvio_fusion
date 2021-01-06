#ifndef lvio_fusion_POSE_GRAPH_H
#define lvio_fusion_POSE_GRAPH_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/frontend.h"

namespace lvio_fusion
{

// [A, B, C]
struct Section
{
    double A = 0; // time of before the first frame
    double B = 0; // time of before the first loop frame
    double C = 0; // the last frame
};

typedef std::map<double, Section> Atlas;

class PoseGraph
{
public:
    typedef std::shared_ptr<PoseGraph> Ptr;

    void SetFrontend(Frontend::Ptr frontend) { frontend_ = frontend; }

    void AddSubMap(double old_time, double start_time, double end_time);

    std::map<double, SE3d> GetActiveSubMaps(Frames &active_kfs, double &old_time, double start_time);

    Atlas GetSections(double start, double end);

    void BuildProblem(Atlas &sections, adapt::Problem &problem);

    void Optimize(Atlas &sections, adapt::Problem &problem);

    void ForwardPropagate(SE3d transfrom, double start_time);

    void ForwardPropagate(SE3d transfrom, const Frames& forward_kfs);

private:
    void UpdateSections(double time);

    Frontend::Ptr frontend_;

    Atlas atlas_;    // loop altas
    Atlas sections_; // sections
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_GRAPH_H
