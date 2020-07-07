#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensors/navsat.h"
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

    Map() {}

    Landmarks &GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return landmarks_;
    }

    Keyframes &GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_;
    }

    Landmarks GetActiveMapPoints(bool full = false);

    Keyframes GetActiveKeyFrames(bool full = false);

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertMapPoint(MapPoint::Ptr mappoint);

    void RemoveMapPoint(MapPoint::Ptr mappoint);

    void Reset()
    {
        landmarks_.clear();
        active_landmarks_.clear();
        keyframes_.clear();
        active_keyframes_.clear();
    }

    Frame::Ptr current_frame = nullptr;
    NavsatMap::Ptr navsat_map;

private:
    void RemoveOldKeyframe();

    std::mutex map_mutex_;
    Landmarks landmarks_;
    Landmarks active_landmarks_;
    Keyframes keyframes_;
    Keyframes active_keyframes_;

    static const int WINDOW_SIZE = 10;
};
} // namespace lvio_fusion

#endif // MAP_H
