#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helper/navsat_error.hpp"
#include "lvio_fusion/ceres_helper/reprojection_error.hpp"
#include "lvio_fusion/ceres_helper/se3_parameterization.hpp"
#include "lvio_fusion/ceres_helper/vehicle_motion_error.hpp"
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
    Map::Landmarks active_landmarks;
    if (active_kfs.empty())
        return;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new SE3dParameterization();

    for (auto kf_pair : active_kfs)
    {
        double *para_kf = kf_pair.second->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        auto left_features = kf_pair.second->left_features;
        for (auto feature_pair : left_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            // freeze points which the frontend is using
            if (landmark->FindLastFrame()->time > (--active_kfs.end())->first)
                continue;
            // clean the map
            if (landmark->observations.size() == 1)
            {
                map_->RemoveMapPoint(landmark);
                continue;
            }
            double *para_landmark = landmark->position.data();
            if (kf_pair.first == active_kfs.begin()->first || kf_pair.first == landmark->FindFirstFrame()->time)
            {
                problem.AddParameterBlock(para_landmark, 3);
                if (landmark->FindFirstFrame()->time < active_kfs.begin()->first)
                {
                    problem.SetParameterBlockConstant(para_landmark);
                }
                else
                {
                    active_landmarks.insert(std::make_pair(landmark->id, landmark));
                }
            }
            ceres::CostFunction *cost_function = new ReprojectionError(to_vector2d(feature->keypoint), left_camera_);
            problem.AddResidualBlock(cost_function, loss_function, para_kf, para_landmark);
        }
        for (auto feature_pair : kf_pair.second->right_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            double *para_landmark = landmark->position.data();
            ceres::CostFunction *cost_function = new ReprojectionError(to_vector2d(feature->keypoint), right_camera_);
            problem.AddResidualBlock(cost_function, loss_function, para_kf, para_landmark);
        }
    }

    if (active_kfs.begin()->first == map_->GetAllKeyFrames().begin()->first)
    {
        auto &pose = active_kfs.begin()->second->pose;
        problem.SetParameterBlockConstant(pose.data());
    }

    // navsat constraints
    auto navsat_map = map_->navsat_map;
    if (map_->navsat_map != nullptr)
    {
        if (navsat_map->initialized)
        {
            double t1 = -1, t2 = -1;
            for (auto kf_pair : active_kfs)
            {
                t2 = kf_pair.first;
                if (t1 != -1)
                {
                    auto navsat_frame = navsat_map->GetFrame(t1, t2);
                    navsat_map->Transfrom(navsat_frame);
                    double *para_kf = kf_pair.second->pose.data();
                    auto p = kf_pair.second->pose.inverse().translation();
                    auto closest = closest_point_on_a_line(navsat_frame.A, navsat_frame.B, p);
                    ceres::CostFunction *cost_function = NavsatError::Create(closest);
                    problem.AddResidualBlock(cost_function, loss_function, para_kf);
                }
                t1 = t2;
            }
        }
        else if (map_->GetAllKeyFrames().size() >= navsat_map->num_frames_init)
        {
            navsat_map->Initialize();
            Optimize(true);
            return;
        }
    }

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

    // reject outliers, never remove the first observation.
    for (auto landmark_pair : active_landmarks)
    {
        auto landmark = landmark_pair.second;
        auto observations = landmark->observations;
        observations.erase(observations.begin());
        for (auto feature_pair : observations)
        {
            auto feature = feature_pair.second;
            auto kf_iter = active_kfs.find(feature->frame.lock()->time);
            if (kf_iter == active_kfs.end())
                continue;
            auto frame = (*kf_iter).second;

            Vector2d error = to_vector2d(feature->keypoint) - left_camera_->world2pixel(landmark->position, frame->pose);
            if (error[0] > 2 || error[1] > 2)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
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
        for (auto feature_pair : kf_pair.second->left_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            // remove lonely points from optimization
            if (landmark->observations.size() == 1)
                continue;
            double *para_landmark = landmark->position.data();
            problem.AddParameterBlock(para_landmark, 3);
            for (auto feature_pair : landmark->observations)
            {
                auto feature = feature_pair.second;
                auto frame = feature->frame.lock();
                if (frame->time > kf_pair.first)
                {
                    continue;
                }
                double *para = frame->pose.data();
                problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);
                if (frame->time <= time)
                {
                    problem.SetParameterBlockConstant(para);
                }
                ceres::CostFunction *cost_function = new ReprojectionError(to_vector2d(feature->keypoint), left_camera_);
                problem.AddResidualBlock(cost_function, loss_function, para, para_landmark);
            }
        }
        for (auto feature_pair : kf_pair.second->right_features)
        {
            auto feature = feature_pair.second;
            auto landmark = feature->mappoint.lock();
            // remove lonely points from optimization
            if (landmark->observations.size() == 1)
                continue;
            double *para = feature->frame.lock()->pose.data();
            double *para_landmark = landmark->position.data();
            ceres::CostFunction *cost_function = new ReprojectionError(to_vector2d(feature->keypoint), right_camera_);
            problem.AddResidualBlock(cost_function, loss_function, para, para_landmark);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 3;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // update map points of last frame
    SE3d base = (--active_kfs.end())->second->pose;
    SE3d tf = base.inverse() * old_base;
    for (auto feature_pair : (--active_kfs.end())->second->right_features)
    {
        auto feature = feature_pair.second;
        auto landmark = feature->mappoint.lock();
        landmark->position = tf * landmark->position;
    }
}

} // namespace lvio_fusion