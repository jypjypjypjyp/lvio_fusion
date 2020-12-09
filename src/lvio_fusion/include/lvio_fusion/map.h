#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;

    static Map &Instance()
    {
        static Map instance;
        return instance;
    }

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

    std::mutex mutex_local_kfs;

private:
    Map() {}
    Map(const Map &);
    Map &operator=(const Map &);

    visual::Landmarks landmarks_;
    Frames keyframes_;
};
} // namespace lvio_fusion

#endif // MAP_H
