#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
namespace lvio_fusion
{

void imu::ReComputeBiasVel(Frames &frames, Frame::Ptr &prior_frame)
{
    adapt::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame = prior_frame;
    Frame::Ptr current_frame;
    bool first = true;
    if (frames.size() > 0)
    {
        for (auto &pair_kf : frames)
        {
            current_frame = pair_kf.second;
            if (!current_frame->is_imu_good || current_frame->preintegration == nullptr)
            {
                last_frame = current_frame;
                continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->Vw.data();
            auto para_bg = current_frame->ImuBias.linearized_bg.data();
            auto para_ba = current_frame->ImuBias.linearized_ba.data();
            problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            problem.SetParameterBlockConstant(para_kf);
            if (last_frame && last_frame->is_imu_good && last_frame->preintegration != nullptr)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data();
                auto para_ba_last = last_frame->ImuBias.linearized_ba.data();
                if (first)
                {
                    problem.AddParameterBlock(para_kf_last, SE3d::num_parameters, local_parameterization);
                    problem.AddParameterBlock(para_v_last, 3);
                    problem.AddParameterBlock(para_bg_last, 3);
                    problem.AddParameterBlock(para_ba_last, 3);
                    problem.SetParameterBlockConstant(para_kf_last);
                    problem.SetParameterBlockConstant(para_v_last);
                    problem.SetParameterBlockConstant(para_bg_last);
                    problem.SetParameterBlockConstant(para_ba_last);
                    first = false;
                }
                ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration);
                problem.AddResidualBlock(ProblemType::IMUError, cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
            }
            last_frame = current_frame;
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 4;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    for (auto &pair_kf : frames)
    {
        auto frame = pair_kf.second;
        if (!frame->preintegration || !frame->last_keyframe || !frame->is_imu_good)
        {
            continue;
        }
        Bias bias(frame->ImuBias.linearized_ba, frame->ImuBias.linearized_bg);
        frame->SetNewBias(bias);
    }
    return;
}

void imu::ReComputeBiasVel(Frames &frames)
{
    adapt::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    if (frames.size() > 0)
    {
        for (auto &pair_kf : frames)
        {
            current_frame = pair_kf.second;
            if (!current_frame->is_imu_good || current_frame->preintegration == nullptr)
            {
                last_frame = current_frame;
                continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->Vw.data();
            auto para_bg = current_frame->ImuBias.linearized_bg.data();
            auto para_ba = current_frame->ImuBias.linearized_ba.data();
            problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            problem.SetParameterBlockConstant(para_kf);
            if (last_frame && last_frame->is_imu_good && last_frame->preintegration != nullptr)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data();
                auto para_ba_last = last_frame->ImuBias.linearized_ba.data();
                ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration);
                problem.AddResidualBlock(ProblemType::IMUError, cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
            }
            last_frame = current_frame;
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 4;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    for (auto &pair_kf : frames)
    {
        auto frame = pair_kf.second;
        if (!frame->preintegration || !frame->last_keyframe || !frame->is_imu_good)
        {
            continue;
        }
        Bias bias(frame->ImuBias.linearized_ba, frame->ImuBias.linearized_bg);
        frame->SetNewBias(bias);
    }
    return;
}

void imu::RePredictVel(Frames &frames, Frame::Ptr &prior_frame)
{
    Frame::Ptr last_key_frame = prior_frame;
    for (auto &frame : frames)
    {
        Frame::Ptr current_key_frame = frame.second;
        Vector3d Gz;
        Gz << 0, 0, -Imu::Get()->G;
        Gz = Imu::Get()->Rwg * Gz;
        double t12 = current_key_frame->preintegration->sum_dt;
        Vector3d twb1 = last_key_frame->GetImuPosition();
        Matrix3d Rwb1 = last_key_frame->GetImuRotation();
        Vector3d Vwb1 = last_key_frame->Vw;

        Matrix3d Rwb2 = normalize_R(Rwb1 * current_key_frame->preintegration->GetDeltaRotation(last_key_frame->GetImuBias()).toRotationMatrix());
        Vector3d twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * current_key_frame->preintegration->GetDeltaPosition(last_key_frame->GetImuBias());
        Vector3d Vwb2 = Vwb1 + t12 * Gz + Rwb1 * current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias());
        current_key_frame->SetVelocity(Vwb2);
        current_key_frame->SetNewBias(last_key_frame->GetImuBias());
        last_key_frame = current_key_frame;
    }
}

bool imu::InertialOptimization(Frames &key_frames, Matrix3d &Rwg, double priorG, double priorA)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    // prior bias
    auto para_gyroBias = key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);

    auto para_accBias = key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);

    // optimize gravity, velocity, bias
    Quaterniond rwg(Rwg);
    SO3d RwgSO3(rwg);
    auto para_rwg = RwgSO3.data();

    ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(para_rwg, SO3d::num_parameters, local_parameterization);
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for (Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame = iter->second;
        if (current_frame->preintegration == nullptr)
        {
            last_frame = current_frame;
            continue;
        }
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_v, 3);

        if (last_frame)
        {
            auto para_v_last = last_frame->Vw.data();
            cost_function = ImuErrorG::Create(current_frame->preintegration, current_frame->pose, last_frame->pose, priorA, priorG);
            problem.AddResidualBlock(cost_function, NULL, para_v_last, para_accBias, para_gyroBias, para_v, para_rwg);
        }
        last_frame = current_frame;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // data recovery
    Quaterniond rwg2(RwgSO3.data()[3], RwgSO3.data()[0], RwgSO3.data()[1], RwgSO3.data()[2]);
    Rwg = rwg2.toRotationMatrix();
    for (int i = 0; i < 3; i++)
    {
        if (std::fabs(para_gyroBias[i]) > 0.1)
        {
            return false;
        }
    }
    Bias bias(para_accBias[0], para_accBias[1], para_accBias[2], para_gyroBias[0], para_gyroBias[1], para_gyroBias[2]);
    Vector3d bg;
    bg << para_gyroBias[0], para_gyroBias[1], para_gyroBias[2];
    for (Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame = iter->second;
        Vector3d dbg = current_frame->GetGyroBias() - bg;
        if (dbg.norm() > 0.01)
        {
            current_frame->SetNewBias(bias);
            current_frame->preintegration->Repropagate(bias.linearized_ba, bias.linearized_bg);
        }
        else
        {
            current_frame->SetNewBias(bias);
        }
    }
    return true;
}

void imu::FullInertialBA(Frames &frames, double priorG, double priorA)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = frames.begin()->first;
    SE3d old_pose = frames.begin()->second->pose;
    for (auto &pair_kf : frames)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto &pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
            problem.AddResidualBlock(cost_function, loss_function, para_kf);
        }
    }

    auto para_gyroBias = frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);
    auto para_accBias = frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for (auto &pair_kf : frames)
    {
        current_frame = pair_kf.second;
        if (!current_frame->is_imu_good || current_frame->preintegration == nullptr)
        {
            last_frame = current_frame;
            continue;
        }
        auto para_kf = current_frame->pose.data();
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->is_imu_good)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            cost_function = ImuErrorInit::Create(current_frame->preintegration, priorA, priorG);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_accBias, para_gyroBias, para_kf, para_v);
        }
        last_frame = current_frame;
    }

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    RecoverData(frames, old_pose, false);
    for (Frames::iterator iter = frames.begin(); iter != frames.end(); iter++)
    {
        current_frame = iter->second;
        if (current_frame->is_imu_good)
        {
            Bias bias(para_accBias[0], para_accBias[1], para_accBias[2], para_gyroBias[0], para_gyroBias[1], para_gyroBias[2]);
            current_frame->SetNewBias(bias);
        }
    }
    return;
}

void imu::RecoverData(Frames &frames, SE3d old_pose, bool set_bias)
{
    SE3d new_pose = frames.begin()->second->pose;
    Vector3d old_pose_translation = old_pose.translation();
    Vector3d old_pose_rotation = R2ypr(old_pose.rotationMatrix());
    Vector3d new_pose_rotation = R2ypr(new_pose.rotationMatrix());
    double y_translation = old_pose_rotation.x() - new_pose_rotation.x();
    Matrix3d translation = ypr2R(Vector3d(y_translation, 0, 0));
    if (std::fabs(std::fabs(old_pose_rotation.y()) - 90) < 1.0 ||
        std::fabs(std::fabs(new_pose_rotation.y()) - 90) < 1.0)
    {
        translation = old_pose.rotationMatrix() * new_pose.inverse().rotationMatrix();
    }
    for (auto &pair_kf : frames)
    {
        auto frame = pair_kf.second;
        if (!frame->preintegration || !frame->last_keyframe || !frame->is_imu_good)
        {
            continue;
        }
        frame->SetPose(translation * frame->pose.rotationMatrix(), translation * (frame->pose.translation() - new_pose.translation()) + old_pose_translation);
        frame->SetVelocity(translation * frame->Vw);
        if (set_bias)
        {
            frame->SetNewBias(frame->GetImuBias());
        }
    }
}

} // namespace lvio_fusion
