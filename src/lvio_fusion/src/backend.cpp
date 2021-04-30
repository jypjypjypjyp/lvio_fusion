#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
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

void Backend::BuildProblem(Frames &active_kfs, adapt::Problem &problem, bool use_imu)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;
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

        // IMU
        if (Imu::Num() && Imu::Get()->initialized && use_imu)
        {
            if (frame->is_imu_good && frame->preintegration != nullptr)
            {
                auto para_v = frame->Vw.data();
                auto para_bg = frame->ImuBias.linearized_bg.data();
                auto para_ba = frame->ImuBias.linearized_ba.data();
                problem.AddParameterBlock(para_v, 3);
                problem.AddParameterBlock(para_ba, 3);
                problem.AddParameterBlock(para_bg, 3);
                set_bias_bound(problem, para_ba, para_bg);
                if (last_frame && last_frame->is_imu_good)
                {
                    auto para_v_last = last_frame->Vw.data();
                    auto para_bg_last = last_frame->ImuBias.linearized_bg.data();
                    auto para_ba_last = last_frame->ImuBias.linearized_ba.data();
                    ceres::CostFunction *cost_function = ImuError::Create(frame->preintegration);
                    problem.AddResidualBlock(ProblemType::ImuError, cost_function, NULL, para_last_kf, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
                }
            }
        }

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
}

double compute_reprojection_error(Vector2d ob, Vector3d pw, SE3d pose, Camera::Ptr camera)
{
    Vector2d error(0, 0);
    PoseOnlyReprojectionError(ob, pw, camera, 1)(pose.data(), error.data());
    return error.norm();
}

void Backend::Optimize()
{
    std::unique_lock<std::mutex> lock(mutex);
    Frames active_kfs = Map::Instance().GetKeyFrames(finished);
    if (active_kfs.empty())
        return;

    double start = active_kfs.begin()->first;
    double end = (--active_kfs.end())->first;
    {
        SE3d old_pose = (--active_kfs.end())->second->pose;
        SE3d start_pose = active_kfs.begin()->second->pose;

        if (Imu::Num() && Imu::Get()->initialized)
        {
            imu::ReComputeBiasVel(active_kfs);
        }

        adapt::Problem problem;
        BuildProblem(active_kfs, problem);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_solver_time_in_seconds = 0.6 * window_size_;
        options.num_threads = num_threads;
        ceres::Solver::Summary summary;
        adapt::Solve(options, &problem, &summary);

        if (Imu::Num() && Imu::Get()->initialized)
        {
            imu::RecoverData(active_kfs, start_pose, true);
        }

        // propagate to the last frame
        SE3d new_pose = (--active_kfs.end())->second->pose;
        SE3d transform = new_pose * old_pose.inverse();
        ForwardPropagate(transform, end + epsilon);
        finished = end + epsilon - window_size_;
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
            if (landmark->observations.size() <= 1 && frame->id != Frame::current_frame_id)
            {
                Map::Instance().RemoveLandmark(landmark);
            }
        }
    }

    if (Lidar::Num() && mapping_)
    {
        Frames mapping_kfs = Map::Instance().GetKeyFrames(start, end - window_size_);
        mapping_->Optimize(mapping_kfs);
    }

    if (Navsat::Num() && Navsat::Get()->initialized)
    {
        std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);
        SE3d old_pose = (--active_kfs.end())->second->pose;
        double navsat_start = Navsat::Get()->Optimize(end);
        double fix_start = Navsat::Get()->QuickFix(start, end);
        navsat_start = std::min(navsat_start, fix_start);
        SE3d new_pose = (--active_kfs.end())->second->pose;
        SE3d transform = new_pose * old_pose.inverse();
        PoseGraph::Instance().ForwardPropagate(transform, end + epsilon, false);
        if (navsat_start && mapping_)
        {
            Frames mapping_kfs = Map::Instance().GetKeyFrames(navsat_start);
            for (auto &pair : mapping_kfs)
            {
                mapping_->ToWorld(pair.second);
            }
        }
    }
}

void Backend::ForwardPropagate(SE3d transform, double time)
{
    std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);
    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Frames active_kfs = Map::Instance().GetKeyFrames(time);
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs[last_frame->time] = last_frame;
    }
    PoseGraph::Instance().Propagate(transform, active_kfs);
    InitializeImu(active_kfs, time);

    adapt::Problem problem;
    BuildProblem(active_kfs, problem, false);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    adapt::Solve(options, &problem, &summary);

    if (Imu::Num() && Imu::Get()->initialized)
    {
        Frame::Ptr frame = Map::Instance().GetKeyFrames(0, active_kfs.begin()->first, 1).begin()->second;
        if (active_kfs.size() > 0)
        {
            imu::RePredictVel(active_kfs, frame);
            imu::ReComputeBiasVel(active_kfs, frame);
        }
        frontend_.lock()->last_keyframe_updated = true;
        if (active_kfs.size() == 0)
        {
            frontend_.lock()->UpdateImu(frame->GetImuBias());
        }
        else
        {
            frontend_.lock()->UpdateImu((--active_kfs.end())->second->GetImuBias());
        }
    }

    frontend_.lock()->UpdateCache();
}

void Backend::InitializeImu(Frames active_kfs, double time)
{
    static double init_time = 0;
    static bool initA = false;
    static bool initB = false;
    static bool initializing = false;
    double priorA = 1e3;
    double priorG = 1e1;
    if (Imu::Num() && initializer_->first_init)
    {
        priorA = 0;
        priorG = 0;
    }
    if (Imu::Num() && Imu::Get()->initialized)
    {
        double dt = 0;
        if (init_time)
            dt = (--active_kfs.end())->second->time - init_time;
        if (dt > 5 && !initA)
        {
            initializer_->need_reinit = true;
            initA = true;
            priorA = 1e4;
            priorG = 1e1;
            frontend_.lock()->status = FrontendStatus::INITIALIZING;
        }
        else if (dt > 15 && !initB)
        {
            initializer_->need_reinit = true;
            initB = true;
            priorA = 0;
            priorG = 0;
            frontend_.lock()->status = FrontendStatus::INITIALIZING;
        }
    }
    Frames frames_init;
    SE3d old_pose;
    SE3d new_pose;
    if (Imu::Num() && (!Imu::Get()->initialized || initializer_->need_reinit))
    {
        frames_init = Map::Instance().GetKeyFrames(0, time, initializer_->num_frames);
        old_pose = (--frames_init.end())->second->pose;

        if (frames_init.size() == initializer_->num_frames &&
            frames_init.begin()->first > frontend_.lock()->valid_imu_time &&
            frames_init.begin()->second->preintegration)
        {
            if (!Imu::Get()->initialized)
            {
                init_time = (--frames_init.end())->second->time;
            }
            initializing = true;
        }
    }

    if (initializing)
    {
        LOG(INFO) << "Initializer Start";
        if (initializer_->Initialize(frames_init, priorA, priorG))
        {
            new_pose = (--frames_init.end())->second->pose;
            SE3d transform = new_pose * old_pose.inverse();
            PoseGraph::Instance().Propagate(transform, active_kfs);
            if (frontend_.lock()->status == FrontendStatus::INITIALIZING)
                frontend_.lock()->status = FrontendStatus::TRACKING_GOOD;
            for (auto kf : active_kfs)
            {
                Frame::Ptr frame = kf.second;
                if (frame->preintegration != nullptr)
                    frame->is_imu_good = true;
            }
            LOG(INFO) << "Initiaclizer Finished";
        }
        else
        {
            LOG(INFO) << "Initiaclizer Failed";
        }
        initializing = false;
    }
    return;
}

} // namespace lvio_fusion