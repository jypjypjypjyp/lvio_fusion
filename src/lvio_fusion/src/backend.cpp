#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/pose_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/manager.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

Backend::Backend(double window_size, bool update_weights)
    : window_size_(window_size), update_weights_(update_weights)
{
    thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    thread_global_ = std::thread(std::bind(&Backend::GlobalLoop, this));
}

void Backend::UpdateMap()
{
    map_update_.notify_one();
}

void Backend::BackendLoop()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(mutex);
        map_update_.wait(lock);
        auto t1 = std::chrono::steady_clock::now();
        Optimize();
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Backend cost time: " << time_used.count() << " seconds.";
    }
}

void Backend::GlobalLoop()
{
    double start = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        auto sections = PoseGraph::Instance().GetSections(start, global_end_);
        if (!sections.empty())
        {
            auto t1 = std::chrono::steady_clock::now();
            // new section has nothing to do with backend's window, so run at the sametime.
            Section new_section = sections.begin()->second;
            start = new_section.C;
            SE3d old_pose = Map::Instance().GetKeyFrame(start)->pose;
            if (Navsat::Num() && Navsat::Get()->initialized)
            {
                Navsat::Get()->Optimize(new_section);
                {
                    // update backend and frontend
                    std::unique_lock<std::mutex> lock(mutex);
                    SE3d new_pose = Map::Instance().GetKeyFrame(start)->pose;
                    SE3d transform = new_pose * old_pose.inverse();
                    PoseGraph::Instance().ForwardUpdate(transform, start + epsilon);
                }
                auto t2 = std::chrono::steady_clock::now();
                auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
                LOG(INFO) << "Global cost time: " << time_used.count() << " seconds.";
            }
        }
        if (Navsat::Num() && Navsat::Get()->initialized)
        {
            // quick fix
            std::unique_lock<std::mutex> lock(mutex);
            SE3d old_pose = Map::Instance().GetKeyFrame(global_end_)->pose;
            Navsat::Get()->QuickFix(start, global_end_);
            SE3d new_pose = Map::Instance().GetKeyFrame(global_end_)->pose;
            SE3d transform = new_pose * old_pose.inverse();
            PoseGraph::Instance().ForwardUpdate(transform, global_end_ + epsilon);
        }
    }
}

void Backend::BuildProblem(Frames &active_kfs, adapt::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;
    double global_end = start_time;
    Frame::Ptr last_frame;
    double *para_last_kf;
    for (auto &pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto &pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            auto type = Camera::Get()->Far(landmark->ToWorld(), frame->pose) ? ProblemType::WeakError : ProblemType::VisualError;
            ceres::CostFunction *cost_function;
            if (first_frame == frame)
            {
                double *para_inv_depth = &landmark->inv_depth;
                problem.AddParameterBlock(para_inv_depth, 1);
                cost_function = TwoCameraReprojectionError::Create(cv2eigen(feature->keypoint.pt), cv2eigen(landmark->first_observation->keypoint.pt), Camera::Get(0), Camera::Get(1), 5 * frame->weights.visual);
                problem.AddResidualBlock(ProblemType::Other, cost_function, loss_function, para_inv_depth);
            }
            else if (first_frame->time < start_time)
            {
                global_end = std::min(first_frame->last_keyframe ? first_frame->last_keyframe->time : 0, global_end);
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint.pt), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(type, cost_function, loss_function, para_kf);
            }
            else
            {
                double *para_fist_kf = first_frame->pose.data();
                double *para_inv_depth = &landmark->inv_depth;
                problem.AddParameterBlock(para_inv_depth, 1);
                // first ob is on right camera; current ob is on left camera;
                cost_function = TwoFrameReprojectionError::Create(cv2eigen(landmark->first_observation->keypoint.pt), cv2eigen(feature->keypoint.pt), Camera::Get(0), Camera::Get(1), frame->weights.visual);
                problem.AddResidualBlock(type, cost_function, loss_function, para_inv_depth, para_fist_kf, para_kf);
            }
        }

        if (Imu::Num() && Imu::Get()->initialized)
        {
            if (frame->good_imu)
            {
                auto para_v = frame->Vw.data();
                auto para_bg = frame->bias.linearized_bg.data();
                auto para_ba = frame->bias.linearized_ba.data();
                problem.AddParameterBlock(para_v, 3);
                problem.AddParameterBlock(para_ba, 3);
                problem.AddParameterBlock(para_bg, 3);
                if (last_frame && last_frame->good_imu)
                {
                    auto para_v_last = last_frame->Vw.data();
                    auto para_bg_last = last_frame->bias.linearized_bg.data();
                    auto para_ba_last = last_frame->bias.linearized_ba.data();
                    ceres::CostFunction *cost_function = ImuError::Create(frame->preintegration);
                    problem.AddResidualBlock(ProblemType::ImuError, cost_function, NULL, para_last_kf, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
                }
            }
        }

        // vehicle constraints
        // if (last_frame)
        // {
        //     ceres::CostFunction *cost_function = VehicleError::Create(frame->time - last_frame->time, 1e4);
        //     problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_last_kf, para_kf);
        // }

        // check if weak constraint
        auto num_types = problem.GetTypes(para_kf);
        if (!num_types[ProblemType::ImuError] && num_types[ProblemType::VisualError] < 10)
        {
            if (last_frame)
            {
                ceres::CostFunction *cost_function = PoseGraphError::Create(last_frame->pose, frame->pose, 100);
                problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_last_kf, para_kf);
            }
            else
            {
                ceres::CostFunction *cost_function = PoseError::Create(frame->pose, 100);
                problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_kf);
            }
        }

        last_frame = frame;
        para_last_kf = para_kf;
    }
    global_end_ = global_end;
}

double compute_reprojection_error(Vector2d ob, Vector3d pw, SE3d pose, Camera::Ptr camera)
{
    Vector2d error(0, 0);
    PoseOnlyReprojectionError(ob, pw, camera, 1)(pose.data(), error.data());
    return error.norm();
}

void Backend::Optimize()
{
    Frames active_kfs = Map::Instance().GetKeyFrames(finished);
    if (active_kfs.empty())
        return;

    double start = active_kfs.begin()->first;
    double end = (--active_kfs.end())->first;
    SE3d old_pose = (--active_kfs.end())->second->pose;
    SE3d start_pose = active_kfs.begin()->second->pose;

    adapt::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_solver_time_in_seconds = (end - start) / active_kfs.size();
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    adapt::Solve(options, &problem, &summary);
    if (Imu::Num() && Imu::Get()->initialized)
    {
        imu::RecoverBias(active_kfs);
    }

    // update frontend
    SE3d new_pose = (--active_kfs.end())->second->pose;
    SE3d transform = new_pose * old_pose.inverse();
    UpdateFrontend(transform, end + epsilon);
    finished = end + epsilon - window_size_;

    if (Lidar::Num() && mapping_)
    {
        Frames mapping_kfs = Map::Instance().GetKeyFrames(start, end - window_size_);
        mapping_->Optimize(mapping_kfs);
    }

    // reject outliers and clean the map
    for (auto &pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        auto features_left = frame->features_left;
        for (auto &pair_feature : features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            if (frame != first_frame && compute_reprojection_error(cv2eigen(feature->keypoint.pt), landmark->ToWorld(), frame->pose, Camera::Get()) > 10)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
        }
    }
}

void Backend::UpdateFrontend(SE3d transform, double time)
{
    // perpare for active kfs
    std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);
    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Frames active_kfs = Map::Instance().GetKeyFrames(time);
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs[last_frame->time] = last_frame;
    }
    PoseGraph::Instance().ForwardUpdate(transform, active_kfs);

    adapt::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 1;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    adapt::Solve(options, &problem, &summary);
    if (Imu::Num() && Imu::Get()->initialized)
    {
        imu::RecoverBias(active_kfs);
    }

    // imu initialization
    if (Imu::Num() && (!Navsat::Num() || (Navsat::Num() && Navsat::Get()->initialized)))
    {
        initializer_->Initialize(frontend_.lock()->init_time, time);
    }
    // update imu
    if (Imu::Num() && Imu::Get()->initialized)
    {
        Frame::Ptr prior_frame = Map::Instance().GetKeyFrames(0, time, 1).begin()->second;
        imu::RePredictVel(active_kfs, prior_frame);
        if (active_kfs.size() == 0)
        {
            frontend_.lock()->UpdateImu(prior_frame->bias);
        }
        else
        {
            frontend_.lock()->UpdateImu((--active_kfs.end())->second->bias);
        }
    }

    // update frontend landmarks and pose
    frontend_.lock()->UpdateCache();
}

} // namespace lvio_fusion