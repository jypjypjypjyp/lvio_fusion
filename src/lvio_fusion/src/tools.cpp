#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{
namespace imu
{

void RePredictVel(Frames &frames, Frame::Ptr &prior_frame)
{
    Frame::Ptr last_frame = prior_frame;
    Vector3d G(0, 0, -Imu::Get()->G);
    Vector3d twb1, twb2, Vwb1, Vwb2;
    Matrix3d Rwb1, Rwb2;
    for (auto &pair : frames)
    {
        Frame::Ptr frame = pair.second;
        double t12 = frame->preintegration->sum_dt;
        twb1 = last_frame->t();
        Rwb1 = last_frame->R();
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
    options.max_solver_time_in_seconds = 0.5;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // data recovery
    Quaterniond rwg2(RwgSO3.data()[3], RwgSO3.data()[0], RwgSO3.data()[1], RwgSO3.data()[2]);
    Rwg = rwg2.toRotationMatrix();
    Bias bias = frames.begin()->second->bias;
    if (bias.linearized_bg.norm() > 0.2)
        return false;
    for (auto &pair : frames)
    {
        Frame::Ptr frame = pair.second;
        frame->SetBias(bias);
        frame->preintegration->Repropagate(bias.linearized_ba, bias.linearized_bg);
    }
    return true;
}

void FullBA(Frames &frames, double prior_a, double prior_g)
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
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        if (frame->good_imu)
        {
            double *para_kf = frame->pose.data();
            double *para_v = frame->Vw.data();
            problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
            if (last_frame && last_frame->good_imu)
            {
                double *para_last_kf = last_frame->pose.data();
                double *para_v_last = last_frame->Vw.data();
                cost_function = ImuInitError::Create(frame->preintegration, prior_a, prior_g);
                problem.AddResidualBlock(cost_function, NULL, para_last_kf, para_v_last, para_ba, para_bg, para_kf, para_v);
            }
        }
        last_frame = frame;
    }

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_solver_time_in_seconds = 0.5;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    RecoverBias(frames);
}

void RecoverBias(Frames &frames)
{
    for (auto &pair : frames)
    {
        auto frame = pair.second;
        frame->SetBias(frame->bias);
    }
}

} // namespace imu
} // namespace lvio_fusion
