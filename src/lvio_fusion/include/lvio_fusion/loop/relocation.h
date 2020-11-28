#ifndef lvio_fusion_RELOCATION_H
#define lvio_fusion_RELOCATION_H

#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/loop/atlas.h"
#include "lvio_fusion/loop/loop_constraint.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.hpp"

#include <DBoW3/DBoW3.h>
#include <DBoW3/Database.h>
#include <DBoW3/Vocabulary.h>
#include <bitset>

namespace lvio_fusion
{

typedef std::bitset<256> BRIEF;

inline cv::Mat brief2mat(BRIEF &brief)
{
    return cv::Mat(1, 32, CV_8U, reinterpret_cast<uchar *>(&brief));
}

inline BRIEF mat2brief(const cv::Mat &mat)
{
    BRIEF brief;
    memcpy(&brief, mat.data, 32);
    return brief;
}

inline std::map<unsigned long, BRIEF> mat2briefs(Frame::Ptr frame)
{
    std::map<unsigned long, BRIEF> briefs;
    int i = 0;
    for (auto pair_feature : frame->features_left)
    {
        briefs[pair_feature.first] = mat2brief(frame->descriptors.row(i));
        i++;
    }
    return briefs;
}

enum class RelocationStatus
{
    RUNNING,
    TO_PAUSE,
    PAUSING
};

class Relocation
{
public:
    typedef std::shared_ptr<Relocation> Ptr;

    Relocation(std::string voc_path);

    void SetCameras(Camera::Ptr left, Camera::Ptr right)
    {
        camera_left_ = left;
        camera_right_ = right;
    }

    void SetLidar(Lidar::Ptr lidar) { lidar_ = lidar; }

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetFeatureAssociation(FeatureAssociation::Ptr association) { association_ = association; }

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void SetFrontend(Frontend::Ptr frontend) { frontend_ = frontend; }

    void SetBackend(Backend::Ptr backend) { backend_ = backend; }

    double head = 0;

private:
    void RelocationLoop();

    void AddKeyFrameIntoVoc(Frame::Ptr frame);

    bool DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame);

    bool Relocate(Frame::Ptr frame, Frame::Ptr old_frame);

    bool RelocateByImage(Frame::Ptr frame, Frame::Ptr old_frame);

    bool RelocateByPoints(Frame::Ptr frame, Frame::Ptr old_frame);

    bool SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id);

    int Hamming(const BRIEF &a, const BRIEF &b);

    void BuildProblem(Frames &active_kfs, ceres::Problem &problem);

    void CorrectLoop(double old_time, double start_time, double end_time);

    void RelocateByPoints(Frames frames);

    DBoW3::Database db_;
    DBoW3::Vocabulary voc_;
    Map::Ptr map_;
    Mapping::Ptr mapping_;
    Frontend::Ptr frontend_;
    Backend::Ptr backend_;
    FeatureAssociation::Ptr association_;
    loop::Atlas atlas_;

    std::thread thread_;
    cv::Ptr<cv::Feature2D> detector_;
    std::map<DBoW3::EntryId, double> map_dbow_to_frames_;

    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;
    Imu::Ptr imu_;
    Lidar::Ptr lidar_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H