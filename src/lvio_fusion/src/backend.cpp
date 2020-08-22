#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

Backend::Backend(double range) : range_(range)
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

void Backend::BuildProblem(Keyframes &active_kfs, ceres::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;

    for (auto kf_pair : active_kfs)
    {
        auto frame = kf_pair.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto feature_pair : frame->features_left)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame();
            ceres::CostFunction *cost_function;
            if (first_frame.lock()->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(feature->keypoint, landmark->ToWorld(), camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else if (first_frame.lock() != frame)
            {
                double *para_fist_kf = first_frame.lock()->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, feature->keypoint, camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
            }
        }
    }

    // navsat constraints
    auto navsat_map = map_->navsat_map;
    if (map_->navsat_map != nullptr && navsat_map->initialized)
    {
        ceres::LossFunction *navsat_loss_function = new ceres::TrivialLoss();
        for (auto kf_pair : active_kfs)
        {
            auto frame = kf_pair.second;
            auto para_kf = frame->pose.data();
            auto np_iter = navsat_map->navsat_points.lower_bound(kf_pair.first);
            auto navsat_point = np_iter->second;
            navsat_map->Transfrom(navsat_point);
            if (std::fabs(navsat_point.time - frame->time) < 1e-2)
            {
                ceres::CostFunction *cost_function = NavsatError::Create(navsat_point.position);
                problem.AddResidualBlock(cost_function, navsat_loss_function, para_kf);
            }
        }
    }

    // lidar constraints
    if (lidar_)
    {
        ceres::LossFunction *lidar_loss_function = new ceres::HuberLoss(1);
        Frame::Ptr last_frame = nullptr;
        Frame::Ptr current_frame = nullptr;
        for (auto kf_pair : active_kfs)
        {
            if (kf_pair.second->feature_lidar)
            {
                current_frame = kf_pair.second;
                if (last_frame)
                {
                    scan_registration_->Associate(current_frame, last_frame, problem, lidar_loss_function);
                }
                last_frame = current_frame;
            }
        }
    }

    if (active_kfs.begin()->first == map_->GetAllKeyFrames().begin()->first)
    {
        auto &pose = active_kfs.begin()->second->pose;
        problem.SetParameterBlockConstant(pose.data());
    }
}

void Backend::Optimize(bool full)
{
    Keyframes active_kfs = map_->GetActiveKeyFrames(full ? 0 : ActiveTime());

    // navsat init
    auto navsat_map = map_->navsat_map;
    if (!full && map_->navsat_map != nullptr && !navsat_map->initialized && map_->GetAllKeyFrames().size() >= navsat_map->num_frames_init)
    {
        navsat_map->Initialize();
        Optimize(true);
        return;
    }

    ceres::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-9;
    options.max_solver_time_in_seconds = range_ * 0.6;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // reject outliers and clean the map
    for (auto kf_pair : active_kfs)
    {
        auto frame = kf_pair.second;
        double *para_kf = frame->pose.data();
        auto left_features = frame->features_left;
        for (auto feature_pair : left_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame();
            Vector2d error(0, 0);
            if (first_frame.lock()->time < active_kfs.begin()->first)
            {
                PoseOnlyReprojectionError(feature->keypoint, landmark->ToWorld(), camera_left_)(para_kf, error.data());
            }
            else if (first_frame.lock() != frame)
            {
                double *para_fist_kf = first_frame.lock()->pose.data();
                TwoFrameReprojectionError(landmark->position, feature->keypoint, camera_left_)(para_fist_kf, para_kf, error.data());
            }
            if (error.norm() > 3)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
            if (landmark->observations.size() == 1 && frame != map_->current_frame)
            {
                map_->RemoveLandmark(landmark);
            }
        }
    }

    // propagate to the last frame
    head_ = (--active_kfs.end())->first;
    Propagate(head_);
}

void Backend::Propagate(double time)
{
    std::unique_lock<std::mutex> lock(frontend_.lock()->last_frame_mutex);

    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Keyframes active_kfs = map_->GetActiveKeyFrames(time);
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs.insert(std::make_pair(last_frame->time, last_frame));
    }

    ceres::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 3;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    frontend_.lock()->UpdateCache();
}

} // namespace lvio_fusion