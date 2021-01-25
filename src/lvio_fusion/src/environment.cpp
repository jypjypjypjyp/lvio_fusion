#include "lvio_fusion/adapt/environment.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"

namespace lvio_fusion
{

std::mutex Environment::mutex;
std::map<double, SE3d> Environment::ground_truths;
std::uniform_real_distribution<double> Environment::u_;
std::vector<Environment::Ptr> Environment::environments_;
Estimator::Ptr Environment::estimator_;
int Environment::num_frames_per_env_ = 10;
bool Environment::initialized_ = false;

SE3d Environment::Optimize()
{
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    Frame::Ptr frame = Frame::Ptr(new Frame());
    frame = state_->second;
    frame->pose = (--Map::Instance().keyframes.find(frame->time))->second->pose;

    // visual
    {
        adapt::Problem problem;
        double *para = frame->pose.data();
        problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        for (auto &pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            ceres::CostFunction *cost_function;
            cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
            problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para);
        }

        // imu
        //TODO

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = num_threads;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    // lidar
    if (estimator_->mapping)
    {
        auto map_frame = Frame::Ptr(new Frame());
        estimator_->mapping->BuildMapFrame(frame, map_frame);
        if (map_frame->feature_lidar && frame->feature_lidar)
        {
            double rpyxyz[6];
            se32rpyxyz(frame->pose * map_frame->pose.inverse(), rpyxyz); // relative_i_j
            if (!map_frame->feature_lidar->points_ground.empty())
            {
                adapt::Problem problem;
                estimator_->association->ScanToMapWithGround(frame, map_frame, rpyxyz, problem);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 2;
                options.num_threads = num_threads;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }
            if (!map_frame->feature_lidar->points_surf.empty())
            {
                adapt::Problem problem;
                estimator_->association->ScanToMapWithSegmented(frame, map_frame, rpyxyz, problem);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 2;
                options.num_threads = num_threads;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }
        }
    }
    
    return frame->pose;
}

inline double compute_reward(SE3d result, SE3d ground_truth)
{
    double rpyxyz_result[6], rpyxyz_ground_truth[6];
    se32rpyxyz(result, rpyxyz_result);
    se32rpyxyz(ground_truth, rpyxyz_ground_truth);
    VectorXd reward;
    reward[0] = rpyxyz_result[0] - rpyxyz_ground_truth[0];
    reward[1] = rpyxyz_result[1] - rpyxyz_ground_truth[1];
    reward[2] = rpyxyz_result[2] - rpyxyz_ground_truth[2];
    reward[3] = rpyxyz_result[3] - rpyxyz_ground_truth[3];
    reward[4] = rpyxyz_result[4] - rpyxyz_ground_truth[4];
    reward[5] = rpyxyz_result[5] - rpyxyz_ground_truth[5];
    return reward.norm();
}

void Environment::Step(Weights *weights, Observation *obs, double *reward, bool *done)
{
    state_->second->weights = *weights;
    SE3d result = Optimize();

    *reward = compute_reward(result, state_->second->pose);

    state_++;
    if (state_ == frames_.end())
    {
        *done = true;
    }
    else
    {
        obs->image = state_->second->image_left;
        obs->points_ground = state_->second->feature_lidar->points_ground;
        obs->points_surf = state_->second->feature_lidar->points_surf;
        *done = false;
    }
}

} // namespace lvio_fusion
