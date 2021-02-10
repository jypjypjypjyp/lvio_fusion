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
    double A = 0;   // for submap: the old time of loop;    for section: the begining of turning
    double B = 0;   // for submap: the begining of loop;    for section: the ending of turning
    double C = 0;   // for submap: ths ending of loop;      for section: the ending of straight line
    SE3d pose;      // temp storage of A's old pose
};

typedef std::map<double, Section> Atlas;

class PoseGraph
{
public:
    typedef std::shared_ptr<PoseGraph> Ptr;

    static PoseGraph &Instance()
    {
        static PoseGraph instance;
        return instance;
    }

    void SetFrontend(Frontend::Ptr frontend) { frontend_ = frontend; }

    Section& AddSubMap(double old_time, double start_time, double end_time);

    Atlas FilterOldSubmaps(double start, double end);

    Atlas GetSections(double start, double end);
    Section GetSection(double time);

    void BuildProblem(Atlas &sections, Section &submap, adapt::Problem &problem);

    void Optimize(Atlas &sections, Section &submap, adapt::Problem &problem);

    void ForwardPropagate(SE3d transfrom, double start_time);

    void Propagate(SE3d transfrom, const Frames& forward_kfs);

    void ForwardPropagate(Section section);
    Atlas sections_;    // sections [A : {A, B, C}]

private:
    PoseGraph() {}
    PoseGraph(const PoseGraph &);
    PoseGraph &operator=(const PoseGraph &);

    void UpdateSections(double time);

    Frontend::Ptr frontend_;

    Atlas submaps_;      // loop submaps [end : {old, start, end}]

    double end_time_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_GRAPH_H
