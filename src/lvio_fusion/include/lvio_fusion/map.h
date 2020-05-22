
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
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;
    typedef std::unordered_map<unsigned long, double *> ParamsType;

    Map() {}

    void InsertKeyFrame(Frame::Ptr frame);
    void InsertMapPoint(MapPoint::Ptr map_point);

    LandmarksType& GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    KeyframesType& GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    LandmarksType& GetActiveMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    KeyframesType& GetActiveKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    ParamsType& GetPoseParams();

    ParamsType& GetPointParams();

    void UpdateMap();

    void CleanMap();

private:
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_;       
    LandmarksType active_landmarks_;
    KeyframesType keyframes_;       
    KeyframesType active_keyframes_;

    std::unordered_map<unsigned long, double *> para_Pose;
    std::unordered_map<unsigned long, double *> para_Point;

    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr first_frame_ = nullptr;

    bool empty_ = true;
    static const int WINDOW_SIZE = 7;
};
} // namespace lvio_fusion

#endif // MAP_H
