#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/manager.h"
#include "lvio_fusion/map.h"
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

void Backend::BuildProblem(Frames &active_kfs, adapt::Problem &problem)
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
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para_kf);
            }
            else if (first_frame != frame)
            {
                double *para_fist_kf = first_frame->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(ProblemType::TwoFrameReprojectionError, cost_function, loss_function, para_fist_kf, para_kf);
            }
        }
    }

    // TODO
    // imu constraints
    // if (imu_ && imu_->initialized)
    // {
    //     Frame::Ptr last_frame;
    //     Frame::Ptr current_frame;
    //     for (auto pair_kf : active_kfs)
    //     {
    //         current_frame = pair_kf.second;
    //         if (!current_frame->preintegration)
    //             continue;
    //         auto para_kf = current_frame->pose.data();
    //         auto para_v = current_frame->preintegration->v0.data();
    //         auto para_ba = current_frame->preintegration->linearized_ba.data();
    //         auto para_bg = current_frame->preintegration->linearized_bg.data();
    //         problem.AddParameterBlock(para_v, 3);
    //         problem.AddParameterBlock(para_ba, 3);
    //         problem.AddParameterBlock(para_bg, 3);
    //         if (last_frame && last_frame->preintegration)
    //         {
    //             auto para_kf_last = last_frame->pose.data();
    //             auto para_v_last = last_frame->preintegration->v0.data();
    //             auto para_ba_last = last_frame->preintegration->linearized_ba.data();
    //             auto para_bg_last = last_frame->preintegration->linearized_bg.data();
    //             ceres::CostFunction *cost_function = ImuError::Create(last_frame->preintegration);
    //             problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
    //         }
    //         last_frame = current_frame;
    //     }
    // }
}

double compute_reprojection_error(Vector2d ob, Vector3d pw, SE3d pose, Camera::Ptr camera)
{
    static double weights[2] = {1, 1};
    Vector2d error(0, 0);
    PoseOnlyReprojectionError(ob, pw, camera, weights)(pose.data(), error.data());
    return error.norm();
}

void Backend::Optimize()
{
    static double forward_head = 0;
    std::unique_lock<std::mutex> lock(mutex);
    Frames active_kfs = Map::Instance().GetKeyFrames(head);

    // TODO: IMU
    // imu init
    // if (imu_ && !initializer_->initialized)
    // {
    //     Frames frames_init = Map::Instance().GetKeyFrames(0, Map::Instance().local_map_head, initializer_->num_frames);
    //     if (frames_init.size() == initializer_->num_frames)
    //     {
    //         imu_->initialized = initializer_->Initialize(frames_init);
    //         frontend_.lock()->status = FrontendStatus::TRACKING_GOOD;
    //     }
    // }

    adapt::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 5;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (mapping_)
    {
        mapping_->Optimize(active_kfs);
    }

    if (Navsat::Num() && Navsat::Get()->initialized)
    {
        double start_time = Navsat::Get()->Optimize((--active_kfs.end())->first);
        if (start_time && mapping_)
        {
            Frames mapping_kfs = Map::Instance().GetKeyFrames(start_time);
            for (auto pair : mapping_kfs)
            {
                mapping_->ToWorld(pair.second);
            }
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
            if (compute_reprojection_error(cv2eigen(feature->keypoint), landmark->ToWorld(), frame->pose, Camera::Get()) > 10)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
            if (landmark->observations.size() == 1 && frame->id != Frame::current_frame_id)
            {
                Map::Instance().RemoveLandmark(landmark);
            }
        }
    }

    // propagate to the last frame
    forward_head = (--active_kfs.end())->first + epsilon;
    ForwardPropagate(forward_head);
    head = forward_head - delay_;
}

void Backend::ForwardPropagate(double time)
{
    std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);

    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Frames active_kfs = Map::Instance().GetKeyFrames(time);
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs[last_frame->time] = last_frame;
    }

    adapt::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    frontend_.lock()->UpdateCache();
}

} // namespace lvio_fusion