
#ifndef MAP_H
#define MAP_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/mappoint.h"

namespace lvio_fusion
{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> Landmarks;
    typedef std::unordered_map<unsigned long, Frame::Ptr> Keyframes;
    typedef std::unordered_map<unsigned long, double *> Params;

    Map() {}

    void InsertKeyFrame(Frame::Ptr frame);
    void InsertMapPoint(MapPoint::Ptr map_point);

    Landmarks& GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    Keyframes& GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    Landmarks& GetActiveMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    Keyframes& GetActiveKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    Params GetPoseParams();

    Params GetPointParams();

    void UpdateMap();

    void CleanMap();

    void Reset()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        landmarks_.clear();
        active_landmarks_.clear();
        keyframes_.clear();
        active_keyframes_.clear();
        empty_ = true;
    }

private:
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    Landmarks landmarks_;       
    Landmarks active_landmarks_;
    Keyframes keyframes_;       
    Keyframes active_keyframes_;

    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr first_frame_ = nullptr;

    bool empty_ = true;
    static const int WINDOW_SIZE = 7;
};
} // namespace lvio_fusion

#endif // MAP_H
