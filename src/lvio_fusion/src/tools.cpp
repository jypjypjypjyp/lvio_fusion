#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{
namespace imu
{

void ReComputeBiasVel(Frames &frames, Frame::Ptr &prior_frame)
{
    adapt::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame = prior_frame;
    bool first = true;
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        if (!frame->good_imu)
        {
            last_frame = frame;
            continue;
        }
        double *para_kf = frame->pose.data();
        double *para_v = frame->Vw.data();
        double *para_bg = frame->bias.linearized_bg.data();
        double *para_ba = frame->bias.linearized_ba.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        problem.AddParameterBlock(para_ba, 3);
        problem.AddParameterBlock(para_bg, 3);
        problem.SetParameterBlockConstant(para_kf);
        if (last_frame && last_frame->good_imu)
        {
            double *para_last_kf = last_frame->pose.data();
            double *para_v_last = last_frame->Vw.data();
            double *para_bg_last = last_frame->bias.linearized_bg.data();
            double *para_ba_last = last_frame->bias.linearized_ba.data();
            if (first)
            {
                problem.AddParameterBlock(para_last_kf, SE3d::num_parameters, local_parameterization);
                problem.AddParameterBlock(para_v_last, 3);
                problem.AddParameterBlock(para_bg_last, 3);
                problem.AddParameterBlock(para_ba_last, 3);
                problem.SetParameterBlockConstant(para_last_kf);
                problem.SetParameterBlockConstant(para_v_last);
                problem.SetParameterBlockConstant(para_bg_last);
                problem.SetParameterBlockConstant(para_ba_last);
                first = false;
            }
            ceres::CostFunction *cost_function = ImuError::Create(frame->preintegration);
            problem.AddResidualBlock(ProblemType::ImuError, cost_function, NULL, para_last_kf, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
        }
        last_frame = frame;
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 4;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        if (frame->good_imu)
        {
            frame->SetBias(frame->bias);
        }
    }
    return;
}

void ReComputeBiasVel(Frames &frames)
{
    adapt::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for (auto &pair : frames)
    {
        current_frame = pair.second;
        if (!current_frame->good_imu)
        {
            last_frame = current_frame;
            continue;
        }
        double *para_kf = current_frame->pose.data();
        double *para_v = current_frame->Vw.data();
        double *para_bg = current_frame->bias.linearized_bg.data();
        double *para_ba = current_frame->bias.linearized_ba.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        problem.AddParameterBlock(para_ba, 3);
        problem.AddParameterBlock(para_bg, 3);
        problem.SetParameterBlockConstant(para_kf);
        if (last_frame && last_frame->good_imu)
        {
            double *para_last_kf = last_frame->pose.data();
            double *para_v_last = last_frame->Vw.data();
            double *para_bg_last = last_frame->bias.linearized_bg.data();
            double *para_ba_last = last_frame->bias.linearized_ba.data();
            ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration);
            problem.AddResidualBlock(ProblemType::ImuError, cost_function, NULL, para_last_kf, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
        }
        last_frame = current_frame;
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 2;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        if (frame->good_imu)
        {
            frame->SetBias(frame->bias);
        }
    }
}

void RePredictVel(Frames &frames, Frame::Ptr &prior_frame)
{
    Frame::Ptr last_frame = prior_frame;
    Vector3d G(0, 0, -Imu::Get()->G);
    G = Imu::Get()->Rwg * G;
    Vector3d twb1, twb2, Vwb1, Vwb2;
    Matrix3d Rwb1, Rwb2;
    for (auto &pair : frames)
    {
        Frame::Ptr frame = pair.second;
        double t12 = frame->preintegration->sum_dt;
        twb1 = last_frame->GetPosition();
        Rwb1 = last_frame->GetRotation();
        Vwb1 = last_frame->Vw;
        Rwb2 = normalize_R(Rwb1 * frame->preintegration->GetDeltaRotation(last_frame->bias).toRotationMatrix());
        twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * G + Rwb1 * frame->preintegration->GetDeltaPosition(last_frame->bias);
        Vwb2 = Vwb1 + t12 * G + Rwb1 * frame->preintegration->GetDeltaVelocity(last_frame->bias);
        frame->SetVelocity(Vwb2);
        frame->SetBias(last_frame->bias);
        last_frame = frame;
    }
}

bool InertialOptimization(Frames &frames, Matrix3d &Rwg, double prior_a, double prior_g)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    // prior bias
    double *para_gyroBias = frames.begin()->second->bias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);
    double *para_accBias = frames.begin()->second->bias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);
    // optimize gravity, velocity, bias
    Quaterniond rwg(Rwg);
    SO3d RwgSO3(rwg);
    double *para_rwg = RwgSO3.data();
    ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(para_rwg, SO3d::num_parameters, local_parameterization);

    Frame::Ptr last_frame;
    for (auto &pair : frames)
    {
        Frame::Ptr frame = pair.second;
        if (!frame->preintegration)
        {
            last_frame = frame;
            continue;
        }
        double *para_v = frame->Vw.data();
        problem.AddParameterBlock(para_v, 3);

        if (last_frame)
        {
            double *para_v_last = last_frame->Vw.data();
            cost_function = ImuInitGError::Create(frame->preintegration, frame->pose, last_frame->pose, prior_a, prior_g);
            problem.AddResidualBlock(cost_function, NULL, para_v_last, para_accBias, para_gyroBias, para_v, para_rwg);
        }
        last_frame = frame;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = num_threads;
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
    Vector3d bg(para_gyroBias[0], para_gyroBias[1], para_gyroBias[2]);
    for (auto &pair : frames)
    {
        Frame::Ptr frame = pair.second;
        Vector3d dbg = frame->bias.linearized_bg - bg;
        frame->SetBias(bias);
        if (dbg.norm() > 0.01)
        {
            frame->preintegration->Repropagate(bias.linearized_ba, bias.linearized_bg);
        }
    }
    return true;
}

void FullInertialBA(Frames &frames, double prior_a, double prior_g)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = frames.begin()->first;
    SE3d old_pose = frames.begin()->second->pose;
    // visual factor
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto &pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            ceres::CostFunction *cost_function;
            if (first_frame == frame)
            {
                double *para_inv_depth = &landmark->inv_depth;
                problem.AddParameterBlock(para_inv_depth, 1);
                cost_function = TwoCameraReprojectionError::Create(cv2eigen(feature->keypoint.pt), cv2eigen(landmark->first_observation->keypoint.pt), Camera::Get(0), Camera::Get(1), 5 * frame->weights.visual);
                problem.AddResidualBlock(cost_function, loss_function, para_inv_depth);
            }
            else if (first_frame->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint.pt), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else
            {
                double *para_fist_kf = first_frame->pose.data();
                double *para_inv_depth = &landmark->inv_depth;
                problem.AddParameterBlock(para_inv_depth, 1);
                // first ob is on right camera; current ob is on left camera;
                cost_function = TwoFrameReprojectionError::Create(cv2eigen(landmark->first_observation->keypoint.pt), cv2eigen(feature->keypoint.pt), Camera::Get(0), Camera::Get(1), frame->weights.visual);
                problem.AddResidualBlock(cost_function, loss_function, para_inv_depth, para_fist_kf, para_kf);
            }
        }
    }

    double *para_bg = frames.begin()->second->bias.linearized_bg.data();
    problem.AddParameterBlock(para_bg, 3);
    double *para_ba = frames.begin()->second->bias.linearized_ba.data();
    problem.AddParameterBlock(para_ba, 3);
    // imu factor
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for (auto &pair : frames)
    {
        current_frame = pair.second;
        if (!current_frame->good_imu)
        {
            last_frame = current_frame;
            continue;
        }
        double *para_kf = current_frame->pose.data();
        double *para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->good_imu)
        {
            double *para_last_kf = last_frame->pose.data();
            double *para_v_last = last_frame->Vw.data();
            cost_function = ImuInitError::Create(current_frame->preintegration, prior_a, prior_g);
            problem.AddResidualBlock(cost_function, NULL, para_last_kf, para_v_last, para_ba, para_bg, para_kf, para_v);
        }
        last_frame = current_frame;
    }

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 2;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    RecoverData(frames, old_pose, false);
    for (auto &pair : frames)
    {
        current_frame = pair.second;
        if (current_frame->good_imu)
        {
            Bias bias(para_ba[0], para_ba[1], para_ba[2], para_bg[0], para_bg[1], para_bg[2]);
            current_frame->SetBias(bias);
        }
    }
}

void RecoverData(Frames &frames, SE3d old_pose, bool set_bias)
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
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        if (!frame->good_imu)
            continue;
        frame->SetPose(translation * frame->pose.rotationMatrix(), translation * (frame->pose.translation() - new_pose.translation()) + old_pose_translation);
        frame->SetVelocity(translation * frame->Vw);
        if (set_bias)
        {
            frame->SetBias(frame->bias);
        }
    }
}

} // namespace imu
} // namespace lvio_fusion
