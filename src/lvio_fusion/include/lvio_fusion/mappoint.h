
#ifndef lvio_fusion_MAPPOINT_H
#define lvio_fusion_MAPPOINT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/semantic/detected_object.h"

namespace lvio_fusion
{

class Frame;

class Feature;

class MapPoint
{
public:
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id = 0; // ID
    bool is_outlier = false;
    int observed_times = 0; // being observed by feature matching algo.
    std::list<std::weak_ptr<Feature>> observations;
    LabelType label = LabelType::None; // Sematic Label

    MapPoint() {}

    MapPoint(long id, Vector3d position);

    Vector3d& Pos()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void SetPos(const Vector3d &pos)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    void AddObservation(std::shared_ptr<Feature> feature)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations.push_back(feature);
        observed_times++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations;
    }

    // factory function
    static MapPoint::Ptr CreateNewMappoint();
private:
    std::mutex data_mutex_;
    Vector3d pos_ = Vector3d::Zero(); // Position in world
};
} // namespace lvio_fusion

#endif // lvio_fusion_MAPPOINT_H
