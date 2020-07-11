#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/sensors/navsat.h"

namespace lvio_fusion
{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> Landmarks;
    typedef std::map<double, Frame::Ptr> Keyframes;

    Map() {}

    Landmarks &GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return landmarks_;
    }

    Keyframes &GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return keyframes_;
    }

    Keyframes GetActiveKeyFrames(bool full = false);

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertMapPoint(MapPoint::Ptr mappoint);

    void RemoveMapPoint(MapPoint::Ptr mappoint);

    void Reset()
    {
        landmarks_.clear();
        keyframes_.clear();
        active_keyframes_.clear();
    }

    Frame::Ptr current_frame = nullptr;
    NavsatMap::Ptr navsat_map;

    static unsigned long current_frame_id;
    static unsigned long current_mappoint_id;

private:
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    Landmarks landmarks_;
    Keyframes keyframes_;
    Keyframes active_keyframes_;

    static const int WINDOW_SIZE = 10;
};
} // namespace lvio_fusion

#endif // MAP_H
