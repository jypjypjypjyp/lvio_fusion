#ifndef lvio_fusion_MAP_H
#define lvio_fusion_MAP_H

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
        return keyframes.size();
    }

    Frame::Ptr GetKeyFrame(double time);
    Frames GetKeyFrames(double start, double end = 0, int num = 0);

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertLandmark(visual::Landmark::Ptr landmark);

    void RemoveLandmark(visual::Landmark::Ptr landmark);

    SE3d ComputePose(double time);

    void Reset()
    {
        landmarks.clear();
        keyframes.clear();
    }
   //IMU
     bool mapUpdated=false;
    void ApplyScaledRotation(const Matrix3d &R);
   //IMUEND
    std::mutex mutex_local_kfs;
    Frames keyframes;
    visual::Landmarks landmarks;
    bool end = false;

private:
    Map() {}
    Map(const Map &);
    Map &operator=(const Map &);
};
} // namespace lvio_fusion

#endif // lvio_fusion_MAP_H
