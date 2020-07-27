#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helpers/navsat_error.hpp"
#include "lvio_fusion/ceres_helpers/pose_only_reprojection_error.hpp"
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
        // Optimize();
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Backend cost time: " << time_used.count() << " seconds.";
    }
}

// void Backend::BuildProblem(Map::Keyframes &active_kfs, ceres::Problem &problem)
// {
//     ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
//     ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
//         new ceres::EigenQuaternionParameterization(),
//         new ceres::IdentityParameterization(3));

//     double start_time = active_kfs.begin()->first;

//     for (auto kf_pair : active_kfs)
//     {
//         auto frame = kf_pair.second;
//         double *para_kf = frame->pose.data();
//         problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
//         for (auto feature_pair : frame->left_features)
//         {
//             auto feature = feature_pair.second;
//             auto landmark = feature->mappoint.lock();
//             auto first_frame = landmark->FindFirstFrame();
//             ceres::CostFunction *cost_function;
//             if (first_frame == frame)
//             {
//                 double *para_depth = &(landmark->depth);
//                 problem.AddParameterBlock(para_depth, 1);
//                 auto init_ob = landmark->init_observation;
//                 cost_function = TwoCameraReprojectionError::Create(feature->keypoint, init_ob->keypoint, left_camera_, right_camera_);
//                 problem.AddResidualBlock(cost_function, loss_function, para_depth);
//             }
//             else if (first_frame->time < start_time)
//             {
//                 // cost_function = PoseOnlyReprojectionError::Create(feature->keypoint, left_camera_, landmark->Position());
//                 // problem.AddResidualBlock(cost_function, loss_function, para_kf);
//             }
//             else
//             {
//                 double *para_depth = &(landmark->depth);
//                 double *para_fist_kf = first_frame->pose.data();
//                 auto init_ob = landmark->observations.begin()->second;
//                 cost_function = TwoFrameReprojectionError::Create(init_ob->keypoint, feature->keypoint, left_camera_);
//                 problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
//             }
//         }
//     }
//     // std::vector<ceres::ResidualBlockId> residual_blocks;
//     // problem.GetResidualBlocksForParameterBlock(para_depth, &residual_blocks);
//     // ceres::Problem::EvaluateOptions EvalOpts;
//     // EvalOpts.num_threads = 4;
//     // EvalOpts.apply_loss_function = false;
//     // EvalOpts.residual_blocks = residual_blocks;
//     // problem.Evaluate(EvalOpts, nullptr, nullptr, nullptr, nullptr);

//     // navsat constraints
//     // auto navsat_map = map_->navsat_map;
//     // if (map_->navsat_map != nullptr && navsat_map->initialized)
//     // {
//     //     ceres::LossFunction *navsat_loss_function = new ceres::TrivialLoss();
//     //     for (auto kf_pair : active_kfs)
//     //     {
//     //         auto frame = kf_pair.second;
//     //         auto para_kf = frame->pose.data();
//     //         auto np_iter = navsat_map->navsat_points.lower_bound(kf_pair.first);
//     //         auto navsat_point = np_iter->second;
//     //         navsat_map->Transfrom(navsat_point);
//     //         if (std::fabs(navsat_point.time - frame->time) < 1e-1)
//     //         {
//     //             ceres::CostFunction *cost_function = NavsatError::Create(navsat_point.position);
//     //             problem.AddResidualBlock(cost_function, navsat_loss_function, para_kf);
//     //         }
//     //         // np_iter--;
//     //         // auto navsat_point1 = np_iter->second;
//     //         // np_iter--;
//     //         // auto navsat_point2 = np_iter->second;
//     //         // auto v1 = (navsat_point.position - navsat_point1.position).normalized();
//     //         // auto v2 = (navsat_point1.position - navsat_point2.position).normalized();
//     //         // if ((v1 - v2).norm() < 1e-2)
//     //         // {
//     //         //     ceres::CostFunction *cost_function = VehicleMotionErrorA::Create(v1);
//     //         //     problem.AddResidualBlock(cost_function, navsat_loss_function, para_kf);
//     //         // }
//     //     }
//     // }

//     // vehicle motion constraints
//     // ceres::LossFunction *vehicle_loss_function = new ceres::TrivialLoss();
//     // Frame::Ptr frame1 = nullptr;
//     // for (auto kf_pair : active_kfs)
//     // {
//     //     auto frame2 = kf_pair.second;
//     //     ceres::CostFunction *cost_function;
//     //     if (frame1 != nullptr)
//     //     {
//     //         double *para1 = frame1->pose.data();
//     //         double *para2 = frame2->pose.data();
//     //         cost_function = VehicleMotionErrorB::Create(frame2->time - frame1->time);
//     //         problem.AddResidualBlock(cost_function, vehicle_loss_function, para1, para2);
//     //         cost_function = VehicleMotionErrorA::Create();
//     //         problem.AddResidualBlock(cost_function, vehicle_loss_function, para2);
//     //     }
//     //     frame1 = frame2;
//     // }

//     if (active_kfs.begin()->first == map_->GetAllKeyFrames().begin()->first)
//     {
//         auto &pose = active_kfs.begin()->second->pose;
//         problem.SetParameterBlockConstant(pose.data());
//     }
// }

// void Backend::Optimize(bool full)
// {
//     Map::Keyframes active_kfs = map_->GetActiveKeyFrames(full);

//     // navsat init
//     // auto navsat_map = map_->navsat_map;
//     // if (!full && map_->navsat_map != nullptr && map_->GetAllKeyFrames().size() >= navsat_map->num_frames_init + navsat_map->epoch * navsat_map->num_frames_epoch)
//     // {
//     //     navsat_map->Initialize();
//     //     Optimize(true);
//     //     navsat_map->epoch++;
//     //     return;
//     // }

//     ceres::Problem problem;
//     BuildProblem(active_kfs, problem);

//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.trust_region_strategy_type = ceres::DOGLEG;
//     options.max_num_iterations = 10;
//     options.num_threads = 4;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     // reject outliers and clean the map
//     for (auto kf_pair : active_kfs)
//     {
//         auto frame = kf_pair.second;
//         double *para_kf = frame->pose.data();
//         auto left_features = frame->left_features;
//         for (auto feature_pair : left_features)
//         {
//             auto feature = feature_pair.second;
//             auto landmark = feature->mappoint.lock();
//             auto first_frame = landmark->FindFirstFrame();
//             Vector2d error(0, 0);
//             if (first_frame != frame)
//             {
//             }
//             else if (first_frame->time < active_kfs.begin()->first)
//             {
//                 // PoseOnlyReprojectionError(feature->keypoint, left_camera_, landmark->Position())(para_kf, error.data());
//             }
//             else
//             {
//                 double *para_depth = &(landmark->depth);
//                 double *para_fist_kf = first_frame->pose.data();
//                 auto init_ob = landmark->observations.begin()->second;
//                 TwoFrameReprojectionError(init_ob->keypoint, feature->keypoint, left_camera_)(para_fist_kf, para_kf, para_depth, error.data());
//             }
//             if (error.norm() > 5)
//             {
//                 landmark->RemoveObservation(feature);
//                 frame->RemoveFeature(feature);
//             }
//             if (landmark->observations.size() == 1 && frame != map_->current_frame)
//             {
//                 map_->RemoveMapPoint(landmark);
//             }
//         }
//     }

//     // propagate to the last frame
//     double end_time = (--active_kfs.end())->first;
//     Propagate(end_time);
// }

// void Backend::Propagate(double time)
// {
//     std::unique_lock<std::mutex> lock(frontend_.lock()->last_frame_mutex);

//     Frame::Ptr last_frame = frontend_.lock()->last_frame;
//     Map::Keyframes &all_kfs = map_->GetAllKeyFrames();
//     Map::Keyframes active_kfs(all_kfs.upper_bound(time), all_kfs.end());
//     if (active_kfs.find(last_frame->time) == active_kfs.end())
//     {
//         active_kfs.insert(std::make_pair(last_frame->time, last_frame));
//     }

//     ceres::Problem problem;
//     BuildProblem(active_kfs, problem);

//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.trust_region_strategy_type = ceres::DOGLEG;
//     options.max_num_iterations = 5;
//     options.num_threads = 8;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     frontend_.lock()->UpdateCache();
// }

} // namespace lvio_fusion