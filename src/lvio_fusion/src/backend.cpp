#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helper/navsat_error.hpp"
#include "lvio_fusion/ceres_helper/reprojection_error.hpp"
#include "lvio_fusion/ceres_helper/se3_parameterization.hpp"
#include "lvio_fusion/ceres_helper/vehicle_motion_error.hpp"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/utility.h"
// #include <boost/format.hpp>

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
        if (map_->GetAllKeyFrames().size() >= epoch * num_frames_epoch)
        {
            Optimize(true);
            epoch++;
        }
        else
        {
            Optimize(true);
        }
        auto t2 = std::chrono::steady_clock::now();
        auto time_used =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Backend cost time: " << time_used.count() << " seconds.";
    }
}

void Backend::Optimize(bool full)
{

    Map::Keyframes active_kfs = map_->GetActiveKeyFrames(full);
    Map::Landmarks active_landmarks = map_->GetActiveMapPoints(full);
    if (active_kfs.empty() || active_landmarks.empty())
        return;
    // std::unique_lock<std::mutex> lock(frontend_.lock()->local_map_mutex);
    // Frame::Ptr last_frame = frontend_.lock()->last_frame;
    // if (active_kfs.find(last_frame->time) == active_kfs.end())
    // {
    //     active_kfs.insert(std::make_pair(last_frame->time, last_frame));
    // }
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new SE3dParameterization();

    for (auto kf_pair : active_kfs)
    {
        double *para = kf_pair.second->pose.data();
        problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);
    }

    for (auto landmark_pair : active_landmarks)
    {
        double *para = landmark_pair.second->position.data();
        problem.AddParameterBlock(para, 3);
        auto observations = landmark_pair.second->GetObservations();
        for (auto feature : observations)
        {
            auto iter = active_kfs.find(feature->frame.lock()->time);
            if (iter == active_kfs.end())
                continue;
            auto keyframe_pair = *iter;
            ceres::CostFunction *cost_function;
            if (feature->is_on_left_image)
            {
                cost_function = new ReprojectionError(to_vector2d(feature->keypoint), camera_left_);
            }
            else
            {
                cost_function = new ReprojectionError(to_vector2d(feature->keypoint), camera_right_);
            }
            double *para_kf = keyframe_pair.second->pose.data();
            problem.AddResidualBlock(cost_function, loss_function, para_kf, para);
        }
    }

    // navsat constraints
    auto navsat_map = map_->navsat_map;
    if (map_->navsat_map != nullptr)
    {
        // if (navsat_map->initialized && full)
        // {
        //     // std::ofstream o1(str(boost::format("/home/jyp/Projects/test-jyp/kps-%1%.csv") % navsat_map->epoch));
        //     // std::ofstream o2(str(boost::format("/home/jyp/Projects/test-jyp/As-%1%.csv") % navsat_map->epoch));
        //     // std::ofstream o3(str(boost::format("/home/jyp/Projects/test-jyp/cps-%1%.csv") % navsat_map->epoch));
        //     double t1 = -1, t2 = -1;
        //     for (auto kf_pair : active_kfs)
        //     {
        //         t2 = kf_pair.first;
        //         if (t1 != -1)
        //         {
        //             auto navsat_frame = navsat_map->GetFrame(t1, t2);
        //             navsat_map->Transfrom(navsat_frame);
        //             double *para_kf = kf_pair.second->pose.data();
        //             auto p = kf_pair.second->pose.inverse().translation();
        //             auto closest = closest_point_on_a_line(navsat_frame.A, navsat_frame.B, p);
        //             // o1 << p[0] << "," << p[1] << "," << p[2] << std::endl;
        //             // o2 << navsat_frame.A[0] << "," << navsat_frame.A[1] << "," << navsat_frame.A[2] << std::endl;
        //             // o3 << closest[0] << "," << closest[1] << "," << closest[2] << std::endl;
        //             ceres::CostFunction *cost_function = Navsat1PointError::Create(closest, para_kf);
        //             problem.AddResidualBlock(cost_function, loss_function, para_kf + 4);
        //         }
        //         t1 = t2;
        //     }
        //     // o1.close();
        //     // o2.close();
        //     // o3.close();
        // }
        // else if (map_->GetAllKeyFrames().size() >= navsat_map->num_frames_init + navsat_map->epoch * navsat_map->num_frames_epoch)
        // {
        //     navsat_map->Initialize();
        //     Optimize(true);
        //     navsat_map->epoch++;
        // }
    }

    // vehicle motion constraints
    if (full)
    {
        for (auto kf_pair : active_kfs)
        {
            double *para = kf_pair.second->pose.data();
            ceres::CostFunction *cost_function = VehicleMotionError::Create();
            problem.AddResidualBlock(cost_function, loss_function, para);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 10;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // if (full)
    // {
    //     std::ofstream o1(str(boost::format("/home/jyp/Projects/test-jyp/kps-%1%.csv") % navsat_map->epoch));
    //     std::ofstream o2(str(boost::format("/home/jyp/Projects/test-jyp/As-%1%.csv") % navsat_map->epoch));
    //     std::ofstream o3(str(boost::format("/home/jyp/Projects/test-jyp/cps-%1%.csv") % navsat_map->epoch));
    //     double t1 = -1, t2 = -1;
    //     for (auto kf_pair : active_kfs)
    //     {
    //         t2 = kf_pair.first;
    //         if (t1 != -1)
    //         {
    //             auto navsat_frame = navsat_map->GetFrame(t1, t2);
    //             navsat_map->Transfrom(navsat_frame);
    //             double *para_kf = para_kfs[kf_pair.second->id];
    //             auto p = kf_pair.second->pose.inverse().translation();
    //             auto closest = closest_point_on_a_line(navsat_frame.A, navsat_frame.B, p);
    //             o1 << p[0] << "," << p[1] << "," << p[2] << std::endl;
    //             o2 << navsat_frame.A[0] << "," << navsat_frame.A[1] << "," << navsat_frame.A[2] << std::endl;
    //             o3 << closest[0] << "," << closest[1] << "," << closest[2] << std::endl;
    //         }
    //         t1 = t2;
    //     }
    //     o1.close();
    //     o2.close();
    //     o3.close();
    // }
    // reject outliers
    for (auto landmark_pair : active_landmarks)
    {
        auto observations = landmark_pair.second->GetObservations();
        for (auto feature : observations)
        {
            auto iter = active_kfs.find(feature->frame.lock()->id);
            if (iter == active_kfs.end())
                continue;
            auto keyframe_pair = (*iter);

            Vector2d error = to_vector2d(feature->keypoint) - camera_left_->world2pixel(landmark_pair.second->position, keyframe_pair.second->pose);
            if (error[0] > 2 || error[1] > 2)
            {
                landmark_pair.second->RemoveObservation(feature);
                feature->frame.lock()->RemoveFeature(feature);
            }
        }
    }

    // lock.unlock();
    // clean the map
    for (auto landmark_pair : active_landmarks)
    {
        if (landmark_pair.second->FindFirstFrame() == landmark_pair.second->FindLastFrame())
        {
            map_->RemoveMapPoint(landmark_pair.second);
        }
    }
    // int current_frame_time = last_frame->time;
    // for (auto it = active_landmarks.begin(); it != active_landmarks.end();)
    // {
    //     Frame::Ptr lf = it->second->FindLastFrame();
    //     if (lf == nullptr || lf->time >= current_frame_time)
    //         continue;
    //     else if (it->second->FindFirstFrame() == it->second->FindLastFrame())
    //     {
    //         map_->RemoveMapPoint(it->second);
    //     }
    // }
    Propagate((--active_kfs.end())->second->time);
    // forward propagate
    // {
    //     std::unique_lock<std::mutex> lock(frontend_.lock()->last_frame_mutex);
    //     Frame::Ptr base_frame = (--active_kfs.end())->second;
    //     Frame::Ptr last_frame = frontend_.lock()->last_frame;
    //     Map::Keyframes &all_kfs = map_->GetAllKeyFrames();
    //     SE3d base = base_frame->pose;
    //     SE3d relative_motion = old_base.inverse() * base;
    //     LOG(INFO) << "**********************" << relative_motion.rotationMatrix() << "xxxxxxxxxxxxxxxx" << relative_motion.translation();
    //     auto first_iter = all_kfs.upper_bound(base_frame->time);
    //     for (auto iter = first_iter; iter != all_kfs.end(); iter++)
    //     {
    //         iter->second->pose = iter->second->pose * relative_motion;
    //         for (auto feature : iter->second->left_features)
    //         {
    //             auto landmark = feature->mappoint.lock();
    //             if (iter == first_iter || landmark->FindFirstFrame()->time == iter->first)
    //             {
    //                 landmark->position = relative_motion.inverse() * landmark->position;
    //             }
    //         }
    //     }
    //     if ((--all_kfs.end())->first != last_frame->time)
    //     {
    //         last_frame->pose = last_frame->pose * relative_motion;
    //     }
    // }
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

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new SE3dParameterization();

    for (auto kf_pair : active_kfs)
    {
        for (auto feature : kf_pair.second->left_features)
        {
            auto landmark = feature->mappoint.lock();
            double *para_landmark = landmark->position.data();
            problem.AddParameterBlock(para_landmark, 3);
            for (auto feature : landmark->observations)
            {
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
                ceres::CostFunction *cost_function;
                if (feature->is_on_left_image)
                {
                    cost_function = new ReprojectionError(to_vector2d(feature->keypoint), camera_left_);
                }
                else
                {
                    cost_function = new ReprojectionError(to_vector2d(feature->keypoint), camera_right_);
                }
                problem.AddResidualBlock(cost_function, loss_function, para, para_landmark);
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 3;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport();
}

} // namespace lvio_fusion