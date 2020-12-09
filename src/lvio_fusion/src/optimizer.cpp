#include "lvio_fusion/loop/pose_graph.h"

namespace lvio_fusion
{

void PoseGraph::AddSubMap(double old_time, double start_time, double end_time)
{
    Submap new_submap;
    new_submap.end_time = end_time;
    new_submap.start_time = start_time;
    new_submap.old_time = old_time;
    altas_[end_time] = new_submap;
}

/**
 * build active submaps and inner submaps
 * @param active_kfs
 * @param old_time      time of the first frame
 * @param start_time    time of the first loop frame
 * @return old frame of inner submaps; key is the first frame's time; value is the pose of the first frame
 */
std::map<double, SE3d> PoseGraph::GetActiveSubMaps(Frames& active_kfs, double& old_time, double start_time)
{
    
    auto start_iter = altas_.lower_bound(old_time);
    auto end_iter = altas_.upper_bound(start_time);
    if (start_iter != altas_.end())
    {
        for (auto iter = start_iter; iter != end_iter; iter++)
        {
            if (iter->second.old_time <= old_time)
            {
                // remove outer submap
                auto new_old_iter = ++active_kfs.find(iter->first);
                old_time = new_old_iter->first;
                active_kfs.erase(active_kfs.begin(), new_old_iter);
            }
            else
            {
                // remove inner submap
                active_kfs.erase(++active_kfs.find(iter->second.old_time), ++active_kfs.find(iter->first));
            }
        }
    }

    std::map<double, SE3d> inner_old_frames;
    Frame::Ptr last_frame;
    for(auto pair_kf : active_kfs)
    {
        if(last_frame && last_frame->id + 1 != pair_kf.second->id)
        {
            inner_old_frames[last_frame->time] = last_frame->pose;
        }
        last_frame = pair_kf.second;
    }
    return inner_old_frames;
}



void PoseGraph::UpdateSections(double time)
{

}

Atlas PoseGraph::GetSections(Frames active_kfs)
{
    if(active_kfs.empty())
        return Atlas();

    double start_time = (active_kfs.begin())->first;
    double end_time = (--active_kfs.end())->first;
    UpdateSections(end_time);

    auto start_iter = altas_.lower_bound(start_time);
    auto end_iter = altas_.upper_bound(end_time);
}

} // namespace lvio_fusion