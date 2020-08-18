#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/navsat/navsat.h"

namespace lvio_fusion
{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;
    
    Map() {}

    visual::Landmarks &GetAllLandmarks()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return landmarks_;
    }

    Keyframes &GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return keyframes_;
    }

    Keyframes GetActiveKeyFrames(double time);

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertLandmark(visual::Landmark::Ptr landmark);

    void RemoveLandmark(visual::Landmark::Ptr landmark);

    SE3d ComputePose(double time);

    void Reset()
    {
        landmarks_.clear();
        keyframes_.clear();
    }

    Frame::Ptr current_frame = nullptr;
    NavsatMap::Ptr navsat_map;
    Point3Cloud simple_map;

private:
    std::mutex data_mutex_;
    visual::Landmarks landmarks_;
    Keyframes keyframes_;

};
} // namespace lvio_fusion

#endif // MAP_H
