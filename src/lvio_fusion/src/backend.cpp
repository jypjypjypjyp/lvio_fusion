#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helper/navsat_error.hpp"
#include "lvio_fusion/ceres_helper/reprojection_error.hpp"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/utility.h"
#include <boost/format.hpp>

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
    Map::Landmarks active_landmarks = map_->GetActiveMapPoints(full);
    Map::Params para_kfs = map_->GetPoseParams(full);
    Map::Params para_landmarks = map_->GetPointParams(full);

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    
    for (auto kf_pair : active_kfs)
    {
        double *para = para_kfs[kf_pair.second->id];
        problem.AddParameterBlock(para, 4, local_parameterization);
        problem.AddParameterBlock(para + 4, 3);
    }
    double *first_para = para_kfs[active_kfs.begin()->second->id];
    problem.SetParameterBlockConstant(first_para);
    problem.SetParameterBlockConstant(first_para + 4);

    for (auto landmark_pair : active_landmarks)
    {
        double *para = para_landmarks[landmark_pair.first];
        problem.AddParameterBlock(para, 3);
        auto observations = landmark_pair.second->GetObservations();
        for (auto feature : observations)
        {
            auto iter = active_kfs.find(feature->frame.lock()->id);
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
            double *para_kf = para_kfs[keyframe_pair.second->id];
            problem.AddResidualBlock(cost_function, loss_function, para_kf, para_kf + 4, para);
        }
    }

    // navsat constraints
    // auto navsat_map = map_->navsat_map;
    // if (map_->navsat_map != nullptr)
    // {
    //     if (navsat_map->initialized && full)
    //     {
    //         std::ofstream o1(str(boost::format("/home/jyp/Projects/test-jyp/kps-%1%.csv") % navsat_map->epoch));
    //         std::ofstream o2(str(boost::format("/home/jyp/Projects/test-jyp/As-%1%.csv") % navsat_map->epoch));
    //         std::ofstream o3(str(boost::format("/home/jyp/Projects/test-jyp/cps-%1%.csv") % navsat_map->epoch));
    //         double t1 = -1, t2 = -1;
    //         for (auto kf_pair : active_kfs)
    //         {
    //             t2 = kf_pair.first;
    //             if (t1 != -1)
    //             {
    //                 auto navsat_frame = navsat_map->GetFrame(t1, t2);
    //                 navsat_map->Transfrom(navsat_frame);
    //                 double *para_kf = para_kfs[kf_pair.second->id];
    //                 auto p = kf_pair.second->pose.inverse().translation();
    //                 auto closest = closest_point_on_a_line(navsat_frame.A, navsat_frame.B, p);
    //                 o1 << p[0] << "," << p[1] << "," << p[2] << std::endl;
    //                 o2 << navsat_frame.A[0] << "," << navsat_frame.A[1] << "," << navsat_frame.A[2] << std::endl;
    //                 o3 << closest[0] << "," << closest[1] << "," << closest[2] << std::endl;
    //                 ceres::CostFunction *cost_function = Navsat1PointError::Create(closest, para_kf);
    //                 problem.AddResidualBlock(cost_function, loss_function, para_kf + 4);
    //             }
    //             t1 = t2;
    //         }
    //         o1.close();
    //         o2.close();
    //         o3.close();
    //     }
    //     else if (map_->GetAllKeyFrames().size() >= navsat_map->num_frames_init + navsat_map->epoch * navsat_map->num_frames_epoch)
    //     {
    //         navsat_map->Initialize();
    //         Optimize(true);
    //         navsat_map->epoch++;
    //     }
    // }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 10;
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
    for (auto landmark : active_landmarks)
    {
        auto observations = landmark.second->GetObservations();
        for (auto feature : observations)
        {
            auto iter = active_kfs.find(feature->frame.lock()->id);
            if (iter == active_kfs.end())
                continue;
            auto keyframe_pair = (*iter);

            Vector2d error = to_vector2d(feature->keypoint) - camera_left_->world2pixel(landmark.second->position, keyframe_pair.second->pose);
            if (error[0] * error[0] + error[1] * error[1] > 9)
            {
                landmark.second->RemoveObservation(feature);
                feature->frame.lock()->RemoveFeature(feature);
            }
        }
    }

    // forward propagate
    Frame::Ptr last
    SE3 cur_last_pose = 
    SE3 movement = active_kfs
}

} // namespace lvio_fusion