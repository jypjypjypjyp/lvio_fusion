#include "lvio_fusion/backend.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/ceres_helper/reprojection_error.hpp"
#include "lvio_fusion/ceres_helper/se3_parameterization.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/mappoint.h"

namespace lvio_fusion
{

Backend::Backend()
{
    backend_running_.store(true);
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();
}

void Backend::Stop()
{
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::BackendLoop()
{
    while (backend_running_.load())
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);
        Optimize();
    }
}

void Backend::Optimize()
{
    Map::ParamsType para_kfs = map_->GetPoseParams();
    Map::ParamsType para_landmarks = map_->GetPointParams();
    Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
    Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();
    for (auto &para : para_kfs)
    {
        problem.AddParameterBlock(para.second, SE3::num_parameters, local_parameterization);
    }

    for (auto &landmark : active_landmarks)
    {
        if (landmark.second->is_outlier_)
            continue;
        double *para = para_landmarks[landmark.first];
        problem.AddParameterBlock(para, 3);
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations)
        {
            if (obs.lock() == nullptr)
                continue;
            auto feature = obs.lock();
            if (feature->is_outlier_ || feature->frame_.lock() == nullptr)
                continue;
            auto frame = feature->frame_.lock();
            auto iter = active_kfs.find(frame->keyframe_id_);
            if (iter == active_kfs.end())
                continue;
            auto keyframe = *iter;

            ceres::CostFunction *cost_function;
            if (feature->is_on_left_image_)
            {
                cost_function = new ReprojectionError(toVec2(feature->position_.pt), camera_left_);
            }
            else
            {
                cost_function = new ReprojectionError(toVec2(feature->position_.pt), camera_right_);
            }
            problem.AddResidualBlock(cost_function, loss_function, para_kfs[keyframe.first], para);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // reject outliers
    for (auto &landmark : active_landmarks)
    {
        if (landmark.second->is_outlier_)
            continue;
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations)
        {
            if (obs.lock() == nullptr)
                continue;
            auto feature = obs.lock();
            if (feature->is_outlier_ || feature->frame_.lock() == nullptr)
                continue;
            auto frame = feature->frame_.lock();
            auto iter = active_kfs.find(frame->keyframe_id_);
            if (iter == active_kfs.end())
                continue;
            auto keyframe = (*iter).second;

            Vec2 error = toVec2(feature->position_.pt) - camera_left_->world2pixel(landmark.second->Pos(), keyframe->Pose());
            if (error[0] * error[0] + error[1] * error[1] > 9)
            {
                landmark.second->RemoveObservation(feature);
            }
        }
    }
}

} // namespace lvio_fusion