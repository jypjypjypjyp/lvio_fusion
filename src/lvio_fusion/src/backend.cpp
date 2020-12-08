#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/frontend.h"
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
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), camera_left_, frame->weights.visual);
                problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para_kf);
            }
            else if (first_frame != frame)
            {
                double *para_fist_kf = first_frame->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), camera_left_, frame->weights.visual);
                problem.AddResidualBlock(ProblemType::TwoFrameReprojectionError, cost_function, loss_function, para_fist_kf, para_kf);
            }
        }
    }

    // navsat constraints
    if (navsat_ != nullptr && navsat_->initialized)
    {
        ceres::LossFunction *navsat_loss_function = new ceres::TrivialLoss();
        for (auto pair_kf : active_kfs)
        {
            auto frame = pair_kf.second;
            if (frame->feature_navsat)
            {
                auto feature = frame->feature_navsat;
                auto para_kf = frame->pose.data();
                ceres::CostFunction *cost_function = NavsatError::Create(feature->Position(), feature->Heading(), feature->A(), feature->B(), feature->C(), frame->weights.navsat);
                problem.AddResidualBlock(ProblemType::TwoFrameReprojectionError, cost_function, navsat_loss_function, para_kf);
            }
        }
    }

//NEWADD
    //  if (imu_ && initializer_->initialized)
    // {
    //     LOG(INFO)<<"BACKEND IMU OPTIMIZER  ===>"<<active_kfs.size();
    //     Frame::Ptr last_frame=NULL;
    //     Frame::Ptr current_frame;
    //     int i=active_kfs.size();
    //     i=i/2;
    //     bool first=true;
    //     for (auto kf_pair : active_kfs)
    //     {
    //         i--;
    //         current_frame = kf_pair.second;
    //         if (!current_frame->bImu||!current_frame->mpLastKeyFrame)
    //         { 
    //              first=false;
    //             last_frame=current_frame;
    //             continue;
    //         }
    //         auto para_kf = current_frame->pose.data();
    //         auto para_v = current_frame->mVw.data();
    //         auto para_bg = current_frame->mImuBias.linearized_bg.data();
    //         auto para_ba = current_frame->mImuBias.linearized_ba.data();

    //         problem.AddParameterBlock(para_v, 3);
    //         problem.AddParameterBlock(para_ba, 3);
    //         problem.AddParameterBlock(para_bg, 3);
    //         //if(i>0)
    //         // {
    //         //     problem.SetParameterBlockConstant(para_v);
    //         //     problem.SetParameterBlockConstant(para_ba);
    //         //     problem.SetParameterBlockConstant(para_bg);
    //         // }
    //         if(first){
    //             first=false;
    //             last_frame = current_frame;
    //             continue;
    //         }
    //         if (last_frame && last_frame->bImu&&last_frame->mpLastKeyFrame)
    //         {
    //             auto para_kf_last = last_frame->pose.data();
    //             auto para_v_last = last_frame->mVw.data();
    //             auto para_bg_last = last_frame->mImuBias.linearized_bg.data();//恢复
    //             auto para_ba_last =last_frame->mImuBias.linearized_ba.data();//恢复
    //            ceres::CostFunction *cost_function = InertialError::Create(current_frame->preintegration,initializer_->mRwg);
    //             problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last,  para_bg_last,para_ba_last, para_kf, para_v);

    //             ceres::CostFunction *cost_function_g = GyroRWError::Create(current_frame->preintegration->C.block<3,3>(9,9).inverse());
    //             problem.AddResidualBlock(cost_function_g, NULL, para_bg_last,para_bg);
    //              ceres::CostFunction *cost_function_a = AccRWError::Create(current_frame->preintegration->C.block<3,3>(12,12).inverse());
    //             problem.AddResidualBlock(cost_function_a, NULL, para_ba_last,para_ba);
                
    //             //  LOG(INFO)<<current_frame->id<<": "<<current_frame->mImuBias.linearized_bg.transpose()<<"  "<<last_frame->mImuBias.linearized_bg.transpose();
    //             // LOG(INFO)<<"ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id;
    //             // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose(); 
    //             // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
    //             // LOG(INFO)<<"     current velocity: "<<current_frame->mVw.transpose();
    //             // LOG(INFO)<<"     last  velocity: "<<last_frame->mVw.transpose();
    //             // LOG(INFO)<<"     current bg: "<<current_frame->mImuBias.linearized_bg.transpose(); 
    //             // LOG(INFO)<<"     last bg: "<<last_frame->mImuBias.linearized_bg.transpose();
    //             //  LOG(INFO)<<"     current ba: "<<current_frame->mImuBias.linearized_ba.transpose(); 
    //             // LOG(INFO)<<"     last ba: "<<last_frame->mImuBias.linearized_ba.transpose();
    //          }
    //         last_frame = current_frame;
    //     }
    // }
    
     if (imu_ && initializer_->initialized)
    {
        LOG(INFO)<<"BACKEND IMU OPTIMIZER  ===>"<<active_kfs.size();
        Frame::Ptr last_frame;
        Frame::Ptr current_frame;
        int i=active_kfs.size();
        for (auto kf_pair : active_kfs)
        {
            i--;
            current_frame = kf_pair.second;
            if (!current_frame->bImu||!current_frame->mpLastKeyFrame)
            {
                last_frame=current_frame;
               
                continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->mVw.data();
            auto para_bg = current_frame->mImuBias.linearized_bg.data();
            auto para_ba = current_frame->mImuBias.linearized_ba.data();
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            // if(i>10)
            // {
            //     problem.SetParameterBlockConstant(para_kf);
            //     problem.SetParameterBlockConstant(para_v);
            //     problem.SetParameterBlockConstant(para_ba);
            //     problem.SetParameterBlockConstant(para_bg);
            // }
            if (last_frame && last_frame->bImu&&last_frame->mpLastKeyFrame)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->mVw.data();
                auto para_bg_last = last_frame->mImuBias.linearized_bg.data();//恢复
                auto para_ba_last =last_frame->mImuBias.linearized_ba.data();//恢复
                ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration,initializer_->mRwg);
                problem.AddResidualBlock(ProblemType::IMUError,cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
                // LOG(INFO)<<"ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id;
                // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose();
                // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
                // LOG(INFO)<<"     current velocity: "<<current_frame->mVw.transpose();
                // LOG(INFO)<<"     last  velocity: "<<last_frame->mVw.transpose();
             }
            last_frame = current_frame;
        }
    }
    
    //NEWADDEND
}

double compute_reprojection_error(Vector2d ob, Vector3d pw, SE3d pose, Camera::Ptr camera)
{
    static double weights[2] = {1, 1};
    Vector2d error(0, 0);
    PoseOnlyReprojectionError(ob, pw, camera, weights)(pose.data(), error.data());
    return error.norm();
}

void Backend::Optimize(bool full)
{
    static double forward_head = 0;
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (!full)
    {
        lock.lock();
    }
    Frames active_kfs = Map::Instance().GetKeyFrames(full ? 0 : head);

  //NEWADD
    // imu init
    if (imu_ && !initializer_->initialized)
    {
        Frames frames_init = Map::Instance().GetKeyFrames(0,head, initializer_->num_frames);
        if (frames_init.size() == initializer_->num_frames)
        {
             std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);
            LOG(INFO)<<"Initialized+++++++++++++++++++++";
            initializer_->InitializeIMU(true);
            frontend_.lock()->status = FrontendStatus::TRACKING_GOOD;
             initializer_->initialized=true;
             LOG(INFO)<<"Initialized---------------------------------";
        }
    }
//NEWADDEND

    adapt::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.6 * delay_;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //NEWADD
    if(imu_&&initializer_->initialized)
    {
        for(auto kf_pair : active_kfs){
            auto frame = kf_pair.second;
            if(!frame->preintegration||!frame->mpLastKeyFrame){
                    continue;
            }
            Bias b(frame->mImuBias.linearized_ba[0],frame->mImuBias.linearized_ba[1],frame->mImuBias.linearized_ba[2],frame->mImuBias.linearized_bg[0],frame->mImuBias.linearized_bg[1],frame->mImuBias.linearized_bg[2]);
            frame->SetNewBias(b);
        }
    }
    //NEWADDEND
    if (!full && lidar_)
    {
        mapping_->Optimize(active_kfs);
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
            if (compute_reprojection_error(cv2eigen(feature->keypoint), landmark->ToWorld(), frame->pose, camera_left_) > 10)
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
    //options.max_num_iterations = 1;
    options.max_solver_time_in_seconds = 0.8 * delay_;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //NEWADD
    if(imu_&&initializer_->initialized)
    {
        for(auto kf_pair : active_kfs){
            auto frame = kf_pair.second;
            if(!frame->preintegration||!frame->mpLastKeyFrame){
                    continue;
            }
            Bias b(frame->mImuBias.linearized_ba[0],frame->mImuBias.linearized_ba[1],frame->mImuBias.linearized_ba[2],frame->mImuBias.linearized_bg[0],frame->mImuBias.linearized_bg[1],frame->mImuBias.linearized_bg[2]);
            frame->SetNewBias(b);
        }
    }
    //NEWADDEND
    frontend_.lock()->UpdateCache();
}

} // namespace lvio_fusion