#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/components/navsat.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/mappoint.h"

namespace lvio_fusion
{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> Landmarks;
    typedef std::map<double, Frame::Ptr> Keyframes;
    typedef std::unordered_map<unsigned long, double *> Params;

    Map() {}

    const Landmarks &GetAllMapPoints()
    {
        return landmarks_;
    }

    const Keyframes &GetAllKeyFrames()
    {
        return keyframes_;
    }

    Landmarks GetActiveMapPoints(bool full = false);

    Keyframes GetActiveKeyFrames(bool full = false);

    Params GetPoseParams(bool full = false);

    Params GetPointParams(bool full = false);

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertMapPoint(MapPoint::Ptr map_point);

    void Reset()
    {
        landmarks_.clear();
        active_landmarks_.clear();
        keyframes_.clear();
        active_keyframes_.clear();
        empty = true;
    }

    Frame::Ptr current_frame = nullptr;
    NavsatMap::Ptr navsat_map;
    bool empty = true;

private:
    void RemoveOldKeyframe();

    Landmarks landmarks_;
    Landmarks active_landmarks_;
    Keyframes keyframes_;
    Keyframes active_keyframes_;

    static const int WINDOW_SIZE = 10;
};
} // namespace lvio_fusion

#endif // MAP_H
