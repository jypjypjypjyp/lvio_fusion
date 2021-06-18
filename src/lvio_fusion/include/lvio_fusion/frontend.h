#ifndef lvio_fusion_FRONTEND_H
#define lvio_fusion_FRONTEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/visual/local_map.h"
#include "lvio_fusion/navigation/global_planner.h"//NAVI
namespace lvio_fusion
{

class Backend;

enum class FrontendStatus
{
    BUILDING,
    INITIALIZING,
    TRACKING,
    LOST
};

class Frontend
{
public:
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe);

    bool AddFrame(Frame::Ptr frame);

    void AddImu(double time, Vector3d acc, Vector3d gyr);

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetGlobalPlanner(Global_planner::Ptr globalplanner ) { globalplanner_ = globalplanner; }//NAVI

    void SetLocalPlanner(Local_planner::Ptr localplanner ) { localplanner_ = localplanner; }//NAVI

    void UpdateCache();

    void UpdateImu(const Bias &bias_);

    std::mutex mutex;
    FrontendStatus status = FrontendStatus::BUILDING;
    Frame::Ptr current_frame;
    Frame::Ptr last_frame;
    Frame::Ptr last_keyframe;
    LocalMap local_map;
    double init_time = 0;

private:
    bool Track();

    void ResetImu();

    void InitFrame();

    int TrackLastFrame();

    int Relocate(Frame::Ptr base_frame);

    void CreateKeyframe();

    bool InitMap();

    int TriangulateNewPoints();

    void Preintegrate();

    void PredictState();

    // data
    std::weak_ptr<Backend> backend_;
    Global_planner::Ptr globalplanner_;//NAVI
    Local_planner::Ptr localplanner_;//NAVI
    std::queue<ImuData> imu_buf_;
    imu::Preintegration::Ptr preintegration_last_kf_; // imu pre integration from last key frame
    SE3d last_frame_pose_cache_;
    SE3d relative_i_j_;
    double dt_ = 0;

    // params
    int num_features_init_;
    int num_features_tracking_bad_;
    int num_features_needed_for_keyframe_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_FRONTEND_H
