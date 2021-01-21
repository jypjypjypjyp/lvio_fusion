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

// #include <DBoW3/DBoW3.h>
// #include <DBoW3/Database.h>
// #include <DBoW3/Vocabulary.h>

namespace lvio_fusion
{

class Relocator
{
public:
    typedef std::shared_ptr<Relocator> Ptr;

    Relocator(std::string voc_path);

    void SetFeatureAssociation(FeatureAssociation::Ptr association) { association_ = association; }

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void SetBackend(Backend::Ptr backend) { backend_ = backend; }

    void SetPoseGraph(PoseGraph::Ptr pose_graph) { pose_graph_ = pose_graph; }

private:
    void DetectorLoop();

    // void AddKeyFrameIntoVoc(Frame::Ptr frame);

    bool DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame);

    bool Relocate(Frame::Ptr frame, Frame::Ptr old_frame);

    bool RelocateByImage(Frame::Ptr frame, Frame::Ptr old_frame);

    bool RelocateByPoints(Frame::Ptr frame, Frame::Ptr old_frame);

    void BuildProblem(Frames &active_kfs, adapt::Problem &problem);

    void BuildProblemWithRelocated(Frames &active_kfs, adapt::Problem &problem);

    void CorrectLoop(double old_time, double start_time, double end_time);

    // DBoW3::Database db_;
    // DBoW3::Vocabulary voc_;
    Mapping::Ptr mapping_;
    Backend::Ptr backend_;
    FeatureAssociation::Ptr association_;
    PoseGraph::Ptr pose_graph_;

    std::thread thread_;
    // std::map<DBoW3::EntryId, double> map_dbow_to_frames_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_DETECTOR_H