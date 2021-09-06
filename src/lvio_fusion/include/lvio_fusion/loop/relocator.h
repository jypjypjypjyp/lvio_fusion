#ifndef lvio_fusion_LOOP_DETECTOR_H
#define lvio_fusion_LOOP_DETECTOR_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/loop/loop.h"
#include "lvio_fusion/loop/pose_graph.h"

namespace lvio_fusion
{

class Relocator
{
public:
    typedef std::shared_ptr<Relocator> Ptr;

    Relocator(int mode, double threshold);

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void SetBackend(Backend::Ptr backend) { backend_ = backend; }

private:
    enum Mode
    {
        None = 0,
        VisualOnly = 1,
        LidarOnly = 2,
        VisualAndLidar = 3
    };

    void DetectorLoop();

    bool DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame);

    bool Relocate(Frame::Ptr frame, Frame::Ptr old_frame);

    bool RelocateByImage(Frame::Ptr frame, Frame::Ptr old_frame);

    bool RelocateByPoints(Frame::Ptr frame, Frame::Ptr old_frame);
    void CorrectLoop(double old_time, double start_time, double end_time);

    void UpdateNewSubmap(Frame::Ptr best_frame, Frames &new_submap_kfs);

    Mapping::Ptr mapping_;
    Backend::Ptr backend_;

    std::thread thread_;
    Mode mode_;
    double threshold_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_DETECTOR_H