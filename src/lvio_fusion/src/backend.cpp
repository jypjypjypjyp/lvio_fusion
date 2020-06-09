#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helper/reprojection_error.hpp"
#include "lvio_fusion/ceres_helper/se3_parameterization.hpp"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/map.h"
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

void Backend::Optimize()
{
    Map::Keyframes active_kfs = map_->GetActiveKeyFrames();
    Map::Landmarks active_landmarks = map_->GetActiveMapPoints();
    Map::Params para_kfs = map_->GetPoseParams();
    Map::Params para_landmarks = map_->GetPointParams();

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();
    for (auto keyframe : active_kfs)
    {
        double *para = para_kfs[keyframe.first];
        problem.AddParameterBlock(para, SE3::num_parameters, local_parameterization);
    }

    for (auto landmark : active_landmarks)
    {
        double *para = para_landmarks[landmark.first];
        problem.AddParameterBlock(para, 3);
        auto observations = landmark.second->GetObservations();
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
            problem.AddResidualBlock(cost_function, loss_function, para_kfs[keyframe_pair.first], para);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

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

            Vector2d error = to_vector2d(feature->keypoint) - camera_left_->world2pixel(landmark.second->Position(), keyframe_pair.second->Pose());
            if (error[0] * error[0] + error[1] * error[1] > 9)
            {
                landmark.second->RemoveObservation(feature);
                feature->frame.lock()->RemoveFeature(feature);
            }
        }
    }
}

} // namespace lvio_fusion