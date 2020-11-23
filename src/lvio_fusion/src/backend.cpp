#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

Backend::Backend(double delay) : delay_(delay)
{
    thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap()
{
    map_update_.notify_one();
}

void Backend::Pause()
{
    if (status == BackendStatus::RUNNING)
    {
        std::unique_lock<std::mutex> lock(pausing_mutex_);
        status = BackendStatus::TO_PAUSE;
        pausing_.wait(lock);
    }
}

void Backend::Continue()
{
    if (status == BackendStatus::PAUSING)
    {
        status = BackendStatus::RUNNING;
        running_.notify_one();
    }
}

void Backend::BackendLoop()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(running_mutex_);
        if (status == BackendStatus::TO_PAUSE)
        {
            status = BackendStatus::PAUSING;
            pausing_.notify_one();
            running_.wait(lock);
        }
        map_update_.wait(lock);
        auto t1 = std::chrono::steady_clock::now();
        Optimize();
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Backend cost time: " << time_used.count() << " seconds.";
    }
}

void Backend::BuildProblem(Frames &active_kfs, ceres::Problem &problem,std::vector<double *> &para_gbs,std::vector<double *> &para_abs)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;

    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            ceres::CostFunction *cost_function;
            if (first_frame->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else if (first_frame != frame)
            {
                double *para_fist_kf = first_frame->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
            }
        }
    }

    // navsat constraints
    auto navsat_map = map_->navsat_map;
    if (map_->navsat_map != nullptr && navsat_map->initialized)
    {
        ceres::LossFunction *navsat_loss_function = new ceres::TrivialLoss();
        for (auto pair_kf : active_kfs)
        {
            auto frame = pair_kf.second;
            auto para_kf = frame->pose.data();
            auto np_iter = navsat_map->navsat_points.lower_bound(pair_kf.first);
            auto navsat_point = np_iter->second;
            navsat_map->Transfrom(navsat_point);
            if (std::fabs(navsat_point.time - frame->time) < 1e-2)
            {
                ceres::CostFunction *cost_function = NavsatError::Create(navsat_point.position);
                problem.AddResidualBlock(cost_function, navsat_loss_function, para_kf);
            }
        }
    }

     if (imu_ && initializer_->initialized)
    {
        para_gbs.reserve(active_kfs.size());
        para_abs.reserve(active_kfs.size());
        Frame::Ptr last_frame;
        Frame::Ptr current_frame;
        int i=0;
        for (auto kf_pair : active_kfs)
        {
            current_frame = kf_pair.second;
            if (!current_frame->preintegration)
            {
                i++;
                continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->mVw.data();
            Vector3d g=current_frame->GetGyroBias();
            auto para_bg = g.data();
            para_gbs[i]=para_bg;
            Vector3d a=current_frame->GetAccBias();
            auto para_ba = a.data();
            para_abs[i]=para_ba;
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            if (last_frame && last_frame->preintegration)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->mVw.data();
                Vector3d g2=last_frame->GetGyroBias();
                auto para_bg_last = g2.data();//恢复
                Vector3d a2=last_frame->GetAccBias();
                auto para_ba_last = a2.data();//恢复
                ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration);
                problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
            }
            last_frame = current_frame;
            i++;
        }

    }
}

double compute_reprojection_error(Vector2d ob, Vector3d pw, SE3d pose, Camera::Ptr camera)
{
    Vector2d error(0, 0);
    PoseOnlyReprojectionError(ob, pw, camera)(pose.data(), error.data());
    return error.norm();
}

void Backend::Optimize(bool full){
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (!full)
    {
        lock.lock();
    }
    Frames active_kfs = map_->GetKeyFrames(full ? 0 : head);

    // imu init
    if (imu_ && !initializer_->initialized)
    {
        Frames frames_init = map_->GetKeyFrames(0,head, initializer_->num_frames);
        if (frames_init.size() == initializer_->num_frames)
        {
            initializer_->InitializeIMU(1e2,1e5,true);
            frontend_.lock()->status = FrontendStatus::TRACKING_GOOD;
             initializer_->initialized=true;
        }
    }

    // navsat init
    auto navsat_map = map_->navsat_map;
    if (!full && map_->navsat_map && !navsat_map->initialized && map_->size() >= navsat_map->num_frames_init)
    {
        navsat_map->Initialize();
        // TODO: full optimize
        Optimize(true);
        return;
    }
std::vector<double *> para_gbs;
std::vector<double *> para_abs;

    ceres::Problem problem;
   BuildProblem(active_kfs, problem,para_gbs,para_abs);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = delay_ * 0.8;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(imu_&&initializer_->initialized)
    {
        int i=0;
        for(auto kf_pair : active_kfs){
            if(!kf_pair.second->preintegration)
                continue;
            Bias b(para_abs[i][0],para_abs[i][1],para_abs[i][2],para_gbs[i][0],para_gbs[i][1],para_gbs[i][2]);
            auto frame = kf_pair.second;
            frame->SetNewBias(b);
            i++;
        }
    }
    // reject outliers and clean the map
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        auto left_features = frame->features_left;
        for (auto pair_feature : left_features)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame();
            if (compute_reprojection_error(cv2eigen(feature->keypoint), landmark->ToWorld(), frame->pose, camera_left_) > 3)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
            if (landmark->observations.size() == 1 && frame->id != Frame::current_frame_id)
            {
                map_->RemoveLandmark(landmark);
            }
        }
    }

    // propagate to the last frame
    double temp_head = (--active_kfs.end())->first + epsilon;
    ForwardPropagate(temp_head);
    head = temp_head - delay_;
}

void Backend::ForwardPropagate(double time)
{
    std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);

    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Frames active_kfs = map_->GetKeyFrames(time);
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs[last_frame->time] = last_frame;
    }
std::vector<double *> para_gbs;
std::vector<double *> para_abs;

    ceres::Problem problem;
  BuildProblem(active_kfs, problem,para_gbs,para_abs);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 3;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(imu_&&initializer_->initialized)
    {
        int i=0;
        for(auto kf_pair : active_kfs){
            if(!kf_pair.second->preintegration)
                continue;
            Bias b(para_abs[i][0],para_abs[i][1],para_abs[i][2],para_gbs[i][0],para_gbs[i][1],para_gbs[i][2]);
            auto frame = kf_pair.second;
            frame->SetNewBias(b);
            i++;
        }
    }
    frontend_.lock()->UpdateCache();
}

} // namespace lvio_fusion