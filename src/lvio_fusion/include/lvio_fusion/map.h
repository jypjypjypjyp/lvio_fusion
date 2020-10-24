#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;

    Map() {}

    int size()
    {
        return keyframes_.size();
    }

    Frames &GetAllKeyFrames()
    {
        return keyframes_;
    }

    Frames GetKeyFrames(double start, double end = 0, int num = 0);

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertLandmark(visual::Landmark::Ptr landmark);

    void RemoveLandmark(visual::Landmark::Ptr landmark);

    SE3d ComputePose(double time);

    void Reset()
    {
        landmarks_.clear();
        keyframes_.clear();
    }

    NavsatMap::Ptr navsat_map;
    double local_map_head = 0;
    std::mutex mutex_all_kfs;
    std::mutex mutex_local_kfs;

private:
    visual::Landmarks landmarks_;
    Frames keyframes_;
};
} // namespace lvio_fusion

#endif // MAP_H
