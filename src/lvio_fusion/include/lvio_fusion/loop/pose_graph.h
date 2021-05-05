#ifndef lvio_fusion_POSE_GRAPH_H
#define lvio_fusion_POSE_GRAPH_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

// [A, B, C]
struct Section
{
    double A = 0; // for submap: the old time of loop;    for section: the begining of turning
    double B = 0; // for submap: the begining of loop;    for section: the ending of turning
    double C = 0; // for submap: ths ending of loop;      for section: the ending of straight line
    SE3d pose;    // temp storage of A's old pose
};

typedef std::map<double, Section> Atlas;

inline double frames_distance(double A, double B)
{
    Vector3d a = Map::Instance().GetKeyFrame(A)->pose.translation(),
             b = Map::Instance().GetKeyFrame(B)->pose.translation();
    return (a - b).norm();
}

class PoseGraph
{
public:
    static PoseGraph &Instance()
    {
        static PoseGraph instance;
        return instance;
    }

    void SetFrontend(Frontend::Ptr frontend) { frontend_ = frontend; }

    Section &AddSubMap(double old_time, double start_time, double end_time);

    Atlas FilterOldSubmaps(double start, double end);

    void UpdateSections(double time);

    Atlas GetSections(double start, double end);
    Section GetSection(double time);

    bool AddSection(double time);

    void BuildProblem(Atlas &sections, Section &submap, adapt::Problem &problem);

    void Optimize(Atlas &sections, Section &submap, adapt::Problem &problem);

    void ForwardUpdate(SE3d transfrom, double start_time, bool need_lock = true);

    void ForwardUpdate(SE3d transfrom, const Frames &forward_kfs);

    Section current_section;
    bool turning = false;

private:
    PoseGraph() {}
    PoseGraph(const PoseGraph &);
    PoseGraph &operator=(const PoseGraph &);

    Frontend::Ptr frontend_;

    Atlas submaps_; // loop submaps [end : {old, start, end}]
    Atlas sections_; // sections [A : {A, B, C}]
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_GRAPH_H
