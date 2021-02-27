#include "lvio_fusion/adapt/environment.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"

namespace lvio_fusion
{

std::mutex Environment::mutex;
std::map<double, SE3d> Environment::ground_truths;
std::uniform_real_distribution<double> Environment::u_;
std::default_random_engine Environment::e_;
std::vector<Environment::Ptr> Environment::environments_;
Estimator::Ptr Environment::estimator_;
int Environment::num_frames_per_env_ = 10;
bool Environment::initialized_ = false;

SE3d Environment::Optimize()
{
    int num_threads = 1;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    Frame::Ptr frame = Frame::Ptr(new Frame());
    *frame = *(state_->second);

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
            problem.AddResidualBlock(ProblemType::VisualError, cost_function, loss_function, para);
        }

        // imu
        Frame::Ptr last_frame = frame->last_keyframe;
        if (frame->bImu && frame->preintegration && last_frame->bImu && last_frame->preintegration)
        {
            auto para_kf = frame->pose.data();
            auto para_v = frame->Vw.data();
            auto para_bg = frame->ImuBias.linearized_bg.data();
            auto para_ba = frame->ImuBias.linearized_ba.data();
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            auto para_bg_last = last_frame->ImuBias.linearized_bg.data();
            auto para_ba_last = last_frame->ImuBias.linearized_ba.data();
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            problem.AddParameterBlock(para_kf_last, 7);
            problem.AddParameterBlock(para_v_last, 3);
            problem.AddParameterBlock(para_ba_last, 3);
            problem.AddParameterBlock(para_bg_last, 3);
            problem.SetParameterBlockConstant(para_v);
            problem.SetParameterBlockConstant(para_ba);
            problem.SetParameterBlockConstant(para_bg);
            problem.SetParameterBlockConstant(para_kf_last);
            problem.SetParameterBlockConstant(para_v_last);
            problem.SetParameterBlockConstant(para_ba_last);
            problem.SetParameterBlockConstant(para_bg_last);
            ceres::CostFunction *cost_function = ImuError::Create(frame->preintegration);
            problem.AddResidualBlock(ProblemType::IMUError, cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para, para_v, para_ba, para_bg);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = num_threads;
        ceres::Solver::Summary summary;
        adapt::Solve(options, &problem, &summary);
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
                options.max_num_iterations = 4;
                options.num_threads = num_threads;
                ceres::Solver::Summary summary;
                adapt::Solve(options, &problem, &summary);
            }
            if (!map_frame->feature_lidar->points_surf.empty())
            {
                adapt::Problem problem;
                estimator_->association->ScanToMapWithSegmented(frame, map_frame, rpyxyz, problem);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.num_threads = num_threads;
                ceres::Solver::Summary summary;
                adapt::Solve(options, &problem, &summary);
            }
        }
    }
    LOG(INFO) << "Weights:" << frame->weights.visual << "," << frame->weights.lidar_ground << "," << frame->weights.lidar_surf;
    return frame->pose;
}

inline double compute_reward(SE3d result, SE3d ground_truth, SE3d base)
{
    SE3d error = ground_truth.inverse() * result;
    SE3d relative = base.inverse() * ground_truth;
    double rpyxyz_error[6], rpyxyz_relative[6];
    se32rpyxyz(error, rpyxyz_error);
    se32rpyxyz(relative, rpyxyz_relative);
    VectorXd relative_error(3);
    // relative_error[0] = rpyxyz_error[0] / rpyxyz_relative[0];
    // relative_error[1] = rpyxyz_error[1] / rpyxyz_relative[1];
    // relative_error[2] = rpyxyz_error[2] / rpyxyz_relative[2];
    relative_error[0] = rpyxyz_error[3] / rpyxyz_relative[3];
    relative_error[1] = rpyxyz_error[4] / rpyxyz_relative[4];
    relative_error[2] = rpyxyz_error[5] / rpyxyz_relative[5];
    return std::min(100.0, 1 / relative_error.norm());
}

void Environment::Step(Weights &weights, Observation &obs, float *reward, bool *done)
{
    state_->second->weights = weights;
    SE3d result = Optimize();
    *reward = compute_reward(result, state_->second->pose, state_->second->last_keyframe->pose);
    LOG(INFO) << *reward;
    state_++;
    if (state_ == frames_.end())
    {
        *done = true;
    }
    else
    {
        obs = state_->second->GetObservation();
        *done = false;
    }
}

} // namespace lvio_fusion
