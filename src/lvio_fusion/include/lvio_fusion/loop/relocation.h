#ifndef lvio_fusion_RELOCATION_H
#define lvio_fusion_RELOCATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/lidar/mapping.h"
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
    for (auto pair : frame->features_left)
    {
        briefs.insert(std::make_pair(pair.first, mat2brief(frame->descriptors.row(i))));
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

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void UpdateMap();

    void Pause();

    void Continue();

    RelocationStatus status = RelocationStatus::RUNNING;

private:
    void RelocationLoop();

    void AddKeyFrameIntoVoc(Frame::Ptr frame);

    bool DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame);

    void CorrectLoop(double start_time, double end_time);

    bool Associate(Frame::Ptr frame, Frame::Ptr &old_frame);

    bool SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id);

    int Hamming(const BRIEF &a, const BRIEF &b);

    DBoW3::Database db_;
    DBoW3::Vocabulary voc_;
    Map::Ptr map_;
    Mapping::Ptr mapping_;

    std::thread thread_;
    std::mutex running_mutex_, pausing_mutex_;
    std::condition_variable running_;
    std::condition_variable pausing_;
    std::condition_variable map_update_;
    cv::Ptr<cv::Feature2D> detector_;
    std::map<DBoW3::EntryId, double> map_db_to_frames_;

    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H