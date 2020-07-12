#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helpers/navsat_error.hpp"
#include "lvio_fusion/ceres_helpers/pose_only_reprojection_error.hpp"
#include "lvio_fusion/ceres_helpers/se3_parameterization.hpp"
#include "lvio_fusion/ceres_helpers/two_camera_reprojection_error.hpp"
#include "lvio_fusion/ceres_helpers/two_frame_reprojection_error.hpp"
#include "lvio_fusion/ceres_helpers/vehicle_motion_error.hpp"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

Backend::Backend()
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
        auto time_used =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Backend cost time: " << time_used.count() << " seconds.";
    }
}

void Backend::Optimize(bool full)
{
    Map::Keyframes active_kfs = map_->GetActiveKeyFrames(full);
    if (active_kfs.empty())
        return;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new SE3dParameterization();

    double start_time = active_kfs.begin()->first;
    double end_time = (--active_kfs.end())->first;

    for (auto kf_pair : active_kfs)
    {
        auto frame = kf_pair.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto feature_pair : frame->left_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            auto first_frame = landmark->FindFirstFrame();
            ceres::CostFunction *cost_function;
            if (first_frame == frame)
            {
                double *para_depth = &(landmark->depth);
                problem.AddParameterBlock(para_depth, 1);
                auto init_ob = landmark->init_observation;
                cost_function = TwoCameraReprojectionError::Create(to_vector2d(feature->keypoint), to_vector2d(init_ob->keypoint), left_camera_, right_camera_);
                problem.AddResidualBlock(cost_function, loss_function, para_depth);
            }
            else if (first_frame->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(to_vector2d(feature->keypoint), left_camera_, landmark);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else
            {
                double *para_depth = &(landmark->depth);
                problem.AddParameterBlock(para_depth, 1);
                double *para_fist_kf = first_frame->pose.data();
                auto init_ob = landmark->observations.begin()->second;
                cost_function = TwoFrameReprojectionError::Create(to_vector2d(feature->keypoint), to_vector2d(init_ob->keypoint), left_camera_);
                problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf, para_depth);
            }
        }
    }

    if (active_kfs.begin()->first == map_->GetAllKeyFrames().begin()->first)
    {
        auto &pose = active_kfs.begin()->second->pose;
        problem.SetParameterBlockConstant(pose.data());
    }

    // navsat constraints
    // auto navsat_map = map_->navsat_map;
    // if (map_->navsat_map != nullptr)
    // {
    //     if (navsat_map->initialized)
    //     {
    //         double t1 = -1, t2 = -1;
    //         for (auto kf_pair : active_kfs)
    //         {
    //             t2 = kf_pair.first;
    //             if (t1 != -1)
    //             {
    //                 auto navsat_frame = navsat_map->GetFrame(t1, t2);
    //                 navsat_map->Transfrom(navsat_frame);
    //                 double *para_kf = kf_pair.second->pose.data();
    //                 auto p = kf_pair.second->pose.inverse().translation();
    //                 auto closest = closest_point_on_a_line(navsat_frame.A, navsat_frame.B, p);
    //                 ceres::CostFunction *cost_function = NavsatError::Create(closest);
    //                 problem.AddResidualBlock(cost_function, loss_function, para_kf);
    //             }
    //             t1 = t2;
    //         }
    //     }
    //     else if (map_->GetAllKeyFrames().size() >= navsat_map->num_frames_init)
    //     {
    //         navsat_map->Initialize();
    //         Optimize(true);
    //         return;
    //     }
    // }

    // vehicle motion constraints
    // if (full)
    // {
    // ceres::LossFunction *vehicle_loss_function = new ceres::TrivialLoss();
    // for (auto kf_pair : active_kfs)
    // {
    //     double *para = kf_pair.second->pose.data();
    //     ceres::CostFunction *cost_function = VehicleMotionError::Create();
    //     problem.AddResidualBlock(cost_function, loss_function, para);
    // }
    // }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 10;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // reject outliers and clean the map
    for (auto kf_pair : active_kfs)
    {
        auto frame = kf_pair.second;
        double *para_kf = frame->pose.data();
        auto left_features = frame->left_features;
        for (auto feature_pair : left_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            auto first_frame = landmark->FindFirstFrame();
            double error[2] = {0, 0};
            if (first_frame != frame)
            {
            }
            else if (first_frame->time < active_kfs.begin()->first)
            {
                PoseOnlyReprojectionError(to_vector2d(feature->keypoint), left_camera_, landmark)(para_kf, error);
            }
            else
            {
                double *para_depth = &(landmark->depth);
                double *para_fist_kf = first_frame->pose.data();
                auto init_ob = landmark->observations.begin()->second;
                TwoFrameReprojectionError(to_vector2d(feature->keypoint), to_vector2d(init_ob->keypoint), left_camera_)(para_fist_kf, para_kf, para_depth, error);
            }
            if (error[0] > 2 || error[1] > 2)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
            if (landmark->observations.size() == 1)
            {
                map_->RemoveMapPoint(landmark);
            }
        }
    }

    // propagate
    Propagate((--active_kfs.end())->second->time);
}

void Backend::Propagate(double time)
{
    std::unique_lock<std::mutex> lock(frontend_.lock()->local_map_mutex);
    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Map::Keyframes &all_kfs = map_->GetAllKeyFrames();
    Map::Keyframes active_kfs(all_kfs.upper_bound(time), all_kfs.end());
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs.insert(std::make_pair(last_frame->time, last_frame));
    }
    SE3d old_base = (--active_kfs.end())->second->pose;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new SE3dParameterization();

    for (auto kf_pair : active_kfs)
    {
        auto frame = kf_pair.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto feature_pair : frame->left_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            auto first_frame = landmark->FindFirstFrame();
            ceres::CostFunction *cost_function;
            if (first_frame == frame)
            {
                double *para_depth = &(landmark->depth);
                problem.AddParameterBlock(para_depth, 1);
                auto init_ob = landmark->init_observation;
                cost_function = TwoCameraReprojectionError::Create(to_vector2d(feature->keypoint), to_vector2d(init_ob->keypoint), left_camera_, right_camera_);
                problem.AddResidualBlock(cost_function, loss_function, para_depth);
            }
            else if (first_frame->time <= time)
            {
                cost_function = PoseOnlyReprojectionError::Create(to_vector2d(feature->keypoint), left_camera_, landmark);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else
            {
                double *para_depth = &(landmark->depth);
                problem.AddParameterBlock(para_depth, 1);
                double *para_fist_kf = first_frame->pose.data();
                auto init_ob = landmark->observations.begin()->second;
                cost_function = TwoFrameReprojectionError::Create(to_vector2d(feature->keypoint), to_vector2d(init_ob->keypoint), left_camera_);
                problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf, para_depth);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 3;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

} // namespace lvio_fusion