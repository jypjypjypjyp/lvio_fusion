#include "lvio_fusion/loop/atlas.h"

namespace lvio_fusion
{
namespace loop
{

void Atlas::AddSubMap(double old_time, double start_time, double end_time)
{
    SubMap new_submap;
    new_submap.end_time = end_time;
    new_submap.start_time = start_time;
    new_submap.old_time = old_time;
    submaps_[end_time] = new_submap;
}

/**
 * build active submaps and inner submaps
 * @param active_kfs
 * @param old_time
 * @param start_time
 * @param end_time
 * @return old frame of inner submaps
 */
std::map<double, SE3d> Atlas::GetActiveSubMaps(Frames& active_kfs, double& old_time, double start_time, double end_time)
{
    
    auto start_submaps = submaps_.lower_bound(old_time);
    auto end_submaps = submaps_.upper_bound(end_time);
    if (start_submaps != submaps_.end())
    {
        for (auto iter = start_submaps; iter != end_submaps; iter++)
        {
            if (iter->second.old_time < old_time)
            {
                // remove outer submap
                active_kfs.erase(active_kfs.begin(), ++active_kfs.find(iter->second.end_time));
                old_time = std::max(old_time, iter->second.end_time);
            }
            else
            {
                // remove inner submap
                active_kfs.erase(++active_kfs.find(iter->second.old_time), ++active_kfs.find(iter->second.end_time));
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
}

} // namespace loop
} // namespace lvio_fusion