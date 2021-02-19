#include "lvio_fusion/imu/imuOptimizer.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
namespace lvio_fusion
{
void ImuOptimizer::ComputeGyroBias(const Frames &frames)
{
    const int N = frames.size();
    std::vector<double> vbx;
    vbx.reserve(N);
    std::vector<double> vby;
    vby.reserve(N);
    std::vector<double> vbz;
    vbz.reserve(N);

    Matrix3d H = Matrix3d::Zero();
    Vector3d grad = Vector3d::Zero();
    bool first = true;
    Frame::Ptr pF1;
    for (auto iter : frames)
    {
        if (first)
        {
            pF1 = iter.second;
            first = false;
            continue;
        }
        Frame::Ptr pF2 = iter.second;
        Matrix3d VisionR = pF1->GetImuRotation().transpose() * pF2->GetImuRotation();
        Matrix3d JRg = pF2->preintegration->jacobian.block<3, 3>(3, 12);
        Matrix3d E = pF2->preintegrationFrame->GetUpdatedDeltaRotation().inverse().toRotationMatrix() * VisionR;
        Vector3d e = LogSO3(E);
        // assert(fabs(pF2->time-pF1->time-pF2->preintegration->dT)<0.01);

        Matrix3d J = -InverseRightJacobianSO3(e) * E.transpose() * JRg;
        grad += J.transpose() * e;
        H += J.transpose() * J;
        pF1 = iter.second;
    }

    Vector3d bg = -H.ldlt().solve(grad);
    LOG(INFO) << bg.transpose();
    for (auto iter : frames)
    {
        Frame::Ptr pF = iter.second;
        pF->ImuBias.linearized_ba = Vector3d::Zero();
        pF->ImuBias.linearized_bg = bg;
        pF->SetNewBias(pF->ImuBias);
    }
    return;
}

void ImuOptimizer::ComputeVelocitiesAccBias(const Frames &frames)
{
    const int N = frames.size();
    const int nVar = 3 * N + 3; // 3 velocities/frame + acc bias
    const int nEqs = 6 * (N - 1);

    MatrixXd J(nEqs, nVar);
    J.setZero();
    VectorXd e(nEqs);
    e.setZero();
    Vector3d g;
    g << 0, 0, -Imu::Get()->G;
    g = Imu::Get()->Rwg * g;
    int i = 0;
    bool first = true;
    Frame::Ptr Frame1;
    Frame::Ptr Frame2;
    for (auto iter : frames)
    {
        if (first)
        {
            Frame1 = iter.second;
            first = false;
            continue;
        }

        Frame2 = iter.second;
        Vector3d twb1 = Frame1->GetImuPosition();
        Vector3d twb2 = Frame2->GetImuPosition();
        Matrix3d Rwb1 = Frame1->GetImuRotation();
        Vector3d dP12 = Frame2->preintegration->GetUpdatedDeltaPosition();
        Vector3d dV12 = Frame2->preintegration->GetUpdatedDeltaVelocity();
        Matrix3d JP12 = Frame2->preintegration->jacobian.block<3, 3>(0, 9);
        Matrix3d JV12 = Frame2->preintegration->jacobian.block<3, 3>(6, 9);
        float t12 = Frame2->preintegration->sum_dt;
        // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
        J.block<3, 3>(6 * i, 3 * i) += Matrix3d::Identity() * t12;
        J.block<3, 3>(6 * i, 3 * N) += Rwb1 * JP12;
        e.block<3, 1>(6 * i, 0) = twb2 - twb1 - 0.5f * g * t12 * t12 - Rwb1 * dP12;
        // Velocity v2=v1+g*t+R1*dV12
        J.block<3, 3>(6 * i + 3, 3 * i) += -Matrix3d::Identity();
        J.block<3, 3>(6 * i + 3, 3 * i + 3) += Matrix3d::Identity();
        J.block<3, 3>(6 * i + 3, 3 * N) -= Rwb1 * JV12;
        e.block<3, 1>(6 * i + 3, 0) = g * t12 + Rwb1 * dV12;
        Frame1 = Frame2;
        i++;
    }

    MatrixXd H = J.transpose() * J;
    MatrixXd B = J.transpose() * e;
    VectorXd x = H.ldlt().solve(B);
    Vector3d ba;
    ba(0) = x(3 * N);
    ba(1) = x(3 * N + 1);
    ba(2) = x(3 * N + 2);

    i = 0;
    for (auto iter : frames)
    {
        Frame::Ptr pF = iter.second;
        pF->preintegration->Repropagate(ba, pF->ImuBias.linearized_bg);
        pF->Vw = x.block<3, 1>(3 * i, 0);
        if (i > 0)
        {
            pF->ImuBias.linearized_ba = ba;
            pF->SetNewBias(pF->ImuBias);
        }
        i++;
    }
    return;
}

void ImuOptimizer::ReComputeBiasVel(Frames &frames, Frame::Ptr &prior_frame)
{
    adapt::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame = prior_frame;
    Frame::Ptr current_frame;
    bool first = true;
    //int n=active_kfs.size();
    if (frames.size() > 0)
        for (auto kf_pair : frames)
        {
            current_frame = kf_pair.second;
            if (!current_frame->bImu || !current_frame->last_keyframe || current_frame->preintegration == nullptr)
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
            if (last_frame && last_frame->bImu && last_frame->last_keyframe && last_frame->preintegration != nullptr)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data(); //恢复
                auto para_ba_last = last_frame->ImuBias.linearized_ba.data(); //恢复
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
                ImuOptimizer::showIMUError(para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg, current_frame->preintegration, current_frame->time - 1.40364e+09 + 8.60223e+07);
                problem.AddResidualBlock(ProblemType::IMUError, cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
            }
            last_frame = current_frame;
        }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 4;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << "ReComputeBiasVel  " << summary.BriefReport();
    LOG(INFO) << summary.num_unsuccessful_steps << ":" << summary.num_successful_steps;
    for (auto kf_pair : frames)
    {
        auto frame = kf_pair.second;
        if (!frame->preintegration || !frame->last_keyframe || !frame->bImu)
        {
            continue;
        }
        Bias bias(frame->ImuBias.linearized_ba[0], frame->ImuBias.linearized_ba[1], frame->ImuBias.linearized_ba[2], frame->ImuBias.linearized_bg[0], frame->ImuBias.linearized_bg[1], frame->ImuBias.linearized_bg[2]);
        frame->SetNewBias(bias);
        // LOG(INFO)<<"opt  TIME: "<<frame->time-1.40364e+09+8.60223e+07<<"    V  "<<frame->Vw.transpose()<<"    R  "<<frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<frame->pose.translation().transpose();
    }
    return;
}

void ImuOptimizer::ReComputeBiasVel(Frames &frames)
{
    adapt::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    //int n=active_kfs.size();
    if (frames.size() > 0)
        for (auto kf_pair : frames)
        {
            current_frame = kf_pair.second;
            if (!current_frame->bImu || !current_frame->last_keyframe || current_frame->preintegration == nullptr)
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
            if (last_frame && last_frame->bImu && last_frame->last_keyframe && last_frame->preintegration != nullptr)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data(); //恢复
                auto para_ba_last = last_frame->ImuBias.linearized_ba.data(); //恢复
                ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration);
                ImuOptimizer::showIMUError(para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg, current_frame->preintegration, current_frame->time - 1.40364e+09 + 8.60223e+07);
                problem.AddResidualBlock(ProblemType::IMUError, cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
            }
            last_frame = current_frame;
        }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 4;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << "ReComputeBiasVel  " << summary.BriefReport();
    LOG(INFO) << summary.num_unsuccessful_steps << ":" << summary.num_successful_steps;
    for (auto kf_pair : frames)
    {
        auto frame = kf_pair.second;
        if (!frame->preintegration || !frame->last_keyframe || !frame->bImu)
        {
            continue;
        }
        Bias bias(frame->ImuBias.linearized_ba[0], frame->ImuBias.linearized_ba[1], frame->ImuBias.linearized_ba[2], frame->ImuBias.linearized_bg[0], frame->ImuBias.linearized_bg[1], frame->ImuBias.linearized_bg[2]);
        frame->SetNewBias(bias);
        // LOG(INFO)<<"opt  TIME: "<<frame->time-1.40364e+09+8.60223e+07<<"    V  "<<frame->Vw.transpose()<<"    R  "<<frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<frame->pose.translation().transpose();
    }
    return;
}

void ImuOptimizer::RePredictVel(Frames &frames, Frame::Ptr &prior_frame)
{
    Frame::Ptr last_key_frame = prior_frame;
    bool first = true;
    for (auto kf : frames)
    {
        Frame::Ptr current_key_frame = kf.second;
        Vector3d Gz;
        Gz << 0, 0, -Imu::Get()->G;
        Gz = Imu::Get()->Rwg * Gz;
        double t12 = current_key_frame->preintegration->sum_dt;
        Vector3d twb1 = last_key_frame->GetImuPosition();
        Matrix3d Rwb1 = last_key_frame->GetImuRotation();
        Vector3d Vwb1 = last_key_frame->Vw;

        Matrix3d Rwb2 = NormalizeRotation(Rwb1 * current_key_frame->preintegration->GetDeltaRotation(last_key_frame->GetImuBias()).toRotationMatrix());
        Vector3d twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * current_key_frame->preintegration->GetDeltaPosition(last_key_frame->GetImuBias());
        Vector3d Vwb2 = Vwb1 + t12 * Gz + Rwb1 * current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias());
        current_key_frame->SetVelocity(Vwb2);
        current_key_frame->SetNewBias(last_key_frame->GetImuBias());
        last_key_frame = current_key_frame;
    }
}

bool ImuOptimizer::InertialOptimization(Frames &key_frames, Eigen::Matrix3d &Rwg, double priorG, double priorA, bool isOptRwg)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    //先验BIAS约束
    auto para_gyroBias = key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);

    auto para_accBias = key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);

    //优化重力、BIAS和速度的边
    Quaterniond rwg(Rwg);
    SO3d RwgSO3(rwg);
    auto para_rwg = RwgSO3.data();

    ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(para_rwg, SO3d::num_parameters, local_parameterization);
    if (!isOptRwg)
    {
        problem.SetParameterBlockConstant(para_rwg);
    }
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for (Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame = iter->second;
        if (!current_frame->last_keyframe || current_frame->preintegration == nullptr)
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
    LOG(INFO) << "InertialOptimization  " << summary.BriefReport();
    LOG(INFO) << summary.num_unsuccessful_steps << ":" << summary.num_successful_steps;
    // std::this_thread::sleep_for(std::chrono::seconds(3));

    //数据恢复
    Quaterniond rwg2(RwgSO3.data()[3], RwgSO3.data()[0], RwgSO3.data()[1], RwgSO3.data()[2]);
    Rwg = rwg2.toRotationMatrix();
    for (int i = 0; i < 3; i++)
    {
        if (abs(para_gyroBias[i]) > 0.1)
        {
            return false;
        }
    }
    Bias bias_(para_accBias[0], para_accBias[1], para_accBias[2], para_gyroBias[0], para_gyroBias[1], para_gyroBias[2]);
    Vector3d bg;
    bg << para_gyroBias[0], para_gyroBias[1], para_gyroBias[2];
    LOG(INFO) << bg.transpose();
    for (Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame = iter->second;
        Vector3d dbg = current_frame->GetGyroBias() - bg;
        if (dbg.norm() > 0.01)
        {
            current_frame->SetNewBias(bias_);
            current_frame->preintegration->Repropagate(bias_.linearized_ba, bias_.linearized_bg);
        }
        else
        {
            current_frame->SetNewBias(bias_);
        }
    }
    return true;
}

void ImuOptimizer::FullInertialBA(Frames &key_frames, double priorG, double priorA)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = key_frames.begin()->first;
    SE3d old_pose = key_frames.begin()->second->pose;
    for (auto pair_kf : key_frames)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
            problem.AddResidualBlock(cost_function, loss_function, para_kf);
        }
    }

    Frame::Ptr pIncKF = key_frames.begin()->second;

    auto para_gyroBias = key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);

    auto para_accBias = key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    //bool first=true;
    for (auto kf_pair : key_frames)
    {
        current_frame = kf_pair.second;
        if (!current_frame->bImu || !current_frame->last_keyframe || current_frame->preintegration == nullptr)
        {
            last_frame = current_frame;
            continue;
        }
        auto para_kf = current_frame->pose.data();
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->bImu)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            // if(first){
            //     problem.SetParameterBlockConstant(para_kf_last);
            //     first=false;
            // }
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
    LOG(INFO) << "FullInertialBA  " << summary.BriefReport();
    LOG(INFO) << summary.num_unsuccessful_steps << ":" << summary.num_successful_steps;
    //LOG(INFO)<<summary.BriefReport();
    //数据恢复
    recoverData(key_frames, old_pose, false);
    for (Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame = iter->second;
        if (current_frame->bImu && current_frame->last_keyframe)
        {
            Bias bias(para_accBias[0], para_accBias[1], para_accBias[2], para_gyroBias[0], para_gyroBias[1], para_gyroBias[2]);
            current_frame->SetNewBias(bias);
        }
    }
    return;
}

void ImuOptimizer::recoverData(Frames active_kfs, SE3d old_pose, bool bias)
{
    SE3d new_pose = active_kfs.begin()->second->pose;
    Vector3d old_pose_translation = old_pose.translation();
    Vector3d old_pose_rotation = R2ypr(old_pose.rotationMatrix());
    Vector3d new_pose_rotation = R2ypr(new_pose.rotationMatrix());
    double y_translation = old_pose_rotation.x() - new_pose_rotation.x();
    Matrix3d translation = ypr2R(Vector3d(y_translation, 0, 0));
    if (abs(abs(old_pose_rotation.y()) - 90) < 1.0 || abs(abs(new_pose_rotation.y()) - 90) < 1.0)
    {
        translation = old_pose.rotationMatrix() * new_pose.inverse().rotationMatrix();
    }
    for (auto kf_pair : active_kfs)
    {
        auto frame = kf_pair.second;
        if (!frame->preintegration || !frame->last_keyframe || !frame->bImu)
        {
            continue;
        }
        frame->SetPose(translation * frame->pose.rotationMatrix(), translation * (frame->pose.translation() - new_pose.translation()) + old_pose_translation);
        frame->SetVelocity(translation * frame->Vw);
        if (bias)
            frame->SetNewBias(frame->GetImuBias());
    }
}

void ImuOptimizer::FullInertialBA(Frames &key_frames)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = key_frames.begin()->first;
    SE3d old_pose = key_frames.begin()->second->pose;
    // for (auto pair_kf : key_frames)
    // {
    //     auto frame = pair_kf.second;
    //     double *para_kf = frame->pose.data();
    //     problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
    //     for (auto pair_feature : frame->features_left)
    //     {
    //         auto feature = pair_feature.second;
    //         auto landmark = feature->landmark.lock();
    //         auto first_frame = landmark->FirstFrame().lock();
    //             cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
    //             problem.AddResidualBlock(cost_function, loss_function, para_kf);
    //     }
    // }

    Frame::Ptr pIncKF = key_frames.begin()->second;

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    // bool first=true;
    for (auto kf_pair : key_frames)
    {
        current_frame = kf_pair.second;
        if (!current_frame->bImu || !current_frame->last_keyframe || current_frame->preintegration == nullptr)
        {
            last_frame = current_frame;
            continue;
        }
        auto para_kf = current_frame->pose.data();
        auto para_v = current_frame->Vw.data();
        auto para_gyroBias = current_frame->ImuBias.linearized_bg.data();
        auto para_accBias = current_frame->ImuBias.linearized_ba.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        problem.AddParameterBlock(para_gyroBias, 3);
        problem.AddParameterBlock(para_accBias, 3);
        problem.SetParameterBlockConstant(para_kf);
        if (last_frame && last_frame->bImu)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            auto para_gyroBias_last = last_frame->ImuBias.linearized_bg.data();
            auto para_accBias_last = last_frame->ImuBias.linearized_ba.data();
            // if(first){
            //     problem.SetParameterBlockConstant(para_kf_last);
            //     first=false;
            // }
            cost_function = ImuError::Create(current_frame->preintegration);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_accBias_last, para_gyroBias_last, para_kf, para_v, para_accBias, para_gyroBias);
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
    LOG(INFO) << "FullInertialBA  " << summary.BriefReport();
    LOG(INFO) << summary.num_unsuccessful_steps << ":" << summary.num_successful_steps;
    //数据恢复

    // recoverData(key_frames,old_pose);
    for (Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame = iter->second;
        if (current_frame->bImu && current_frame->last_keyframe)
        {
            current_frame->SetNewBias(current_frame->GetImuBias());
        }
    }
    return;
}

void ImuOptimizer::showIMUError(const double *parameters0, const double *parameters1, const double *parameters2, const double *parameters3, const double *parameters4, const double *parameters5, const double *parameters6, const double *parameters7, imu::Preintegration::Ptr preintegration_, double time)
{
    Quaterniond Qi(parameters0[3], parameters0[0], parameters0[1], parameters0[2]);
    Vector3d Pi(parameters0[4], parameters0[5], parameters0[6]);

    Vector3d Vi(parameters1[0], parameters1[1], parameters1[2]);
    Vector3d Bai(parameters2[0], parameters2[1], parameters2[2]);
    Vector3d Bgi(parameters3[0], parameters3[1], parameters3[2]);

    Quaterniond Qj(parameters4[3], parameters4[0], parameters4[1], parameters4[2]);
    Vector3d Pj(parameters4[4], parameters4[5], parameters4[6]);

    Vector3d Vj(parameters5[0], parameters5[1], parameters5[2]);
    Vector3d Baj(parameters6[0], parameters6[1], parameters6[2]);
    Vector3d Bgj(parameters7[0], parameters7[1], parameters7[2]);
    Matrix<double, 15, 1> residual;
    residual = preintegration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
    Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(preintegration_->covariance.inverse()).matrixL().transpose();

    residual = sqrt_info * residual;
    if (time < 350)
        //LOG(INFO)<<"time"<<time<<"   residual  "<<residual.transpose()<<"  bias  "<<Bai.transpose()<<" "<<Bgi.transpose();
        return;
}

} // namespace lvio_fusion
