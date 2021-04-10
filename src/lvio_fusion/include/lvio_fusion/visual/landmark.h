#ifndef lvio_fusion_LANDMARK_H
#define lvio_fusion_LANDMARK_H

#include "lvio_fusion/visual/feature.h"

namespace lvio_fusion
{
    
namespace visual
{
class Landmark
{
public:
    typedef std::shared_ptr<Landmark> Ptr;

    Vector3d ToWorld();

    void Clear();
    
    std::weak_ptr<Frame> FirstFrame();
    std::weak_ptr<Frame> LastFrame();

    void AddObservation(Feature::Ptr feature);

    void RemoveObservation(Feature::Ptr feature);

    static Landmark::Ptr Create(double depth);

    static unsigned long current_landmark_id;
    unsigned long id = 0;                   // ID
    double depth;                           // depth in the first observation
    Features observations;                  // only for left feature
    Feature::Ptr first_observation;         // the first right observation

private:
    Landmark()
    {
        id = ++current_landmark_id;
    }
};

typedef std::unordered_map<unsigned long, Landmark::Ptr> Landmarks;
} // namespace visual

} // namespace lvio_fusion

#endif // lvio_fusion_LANDMARK_H
