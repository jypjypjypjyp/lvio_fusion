#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/pose_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void Navsat::AddPoint(double time, double x, double y, double z, Vector3d cov)
{
    raw[time] = Vector3d(x, y, z);

    static double finished = 0;
    Frames new_kfs = Map::Instance().GetKeyFrames(finished);
    for (auto &pair : new_kfs)
    {
        auto this_iter = raw.lower_bound(pair.first);
        auto last_iter = this_iter;
        last_iter--;
        if (this_iter == raw.begin() || this_iter == raw.end() || std::fabs(this_iter->first - pair.first) > 1)
            continue;

        double t1 = pair.first - last_iter->first,
               t2 = this_iter->first - pair.first;
        auto p = (this_iter->second * t1 + last_iter->second * t2) / (t1 + t2);
        raw[pair.first] = p;
        pair.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(pair.first, cov));
        finished = pair.first + epsilon;
    }
    if (!initialized && !Map::Instance().keyframes.empty() && frames_distance(0, -1) > min_distance_fix_)
    {
        Initialize();
    }
}

Vector3d Navsat::GetRawPoint(double time)
{
    assert(raw.find(time) != raw.end());
    return raw[time];
}

Vector3d Navsat::GetFixPoint(Frame::Ptr frame)
{
    if (frame->loop_closure)
    {
        return (frame->loop_closure->frame_old->pose * frame->loop_closure->relative_o_c).translation();
    }
    if (frame->feature_navsat)
    {
        return fix + GetPoint(frame->feature_navsat->time);
    }
    else
    {
        return fix + GetAroundPoint(frame->time);
    }
}

Vector3d Navsat::GetPoint(double time)
{
    return extrinsic * GetRawPoint(time);
}

Vector3d Navsat::GetAroundPoint(double time)
{
    if (raw.empty())
        return Vector3d::Zero();
    auto iter = raw.lower_bound(time);
    if (iter == raw.end())
        iter--;
    return extrinsic * iter->second;
}

// make sure time > begin
bool Navsat::EstimatePose(double time, SE3d &pose)
{
    auto iter1 = raw.lower_bound(time);
    auto iter2 = iter1--;
    if (iter2 != raw.begin() && iter2 != raw.end())
    {
        Vector3d p1 = extrinsic * iter1->second;
        Vector3d p2 = extrinsic * iter2->second;
        while (iter1 != raw.begin())
        {
            if ((p1 - p2).norm() > min_distance_fix_)
            {
                pose = get_pose_from_two_points(p1, p2);
                return true;
            }
            else
            {
                iter1--;
                p1 = extrinsic * iter1->second;
            }
        }
    }
    return false;
}

void Navsat::Initialize()
{
    Frames keyframes = Map::Instance().keyframes;

    ceres::Problem problem;
    double para[6] = {0, 0, 0, 0, 0, 0};
    problem.AddParameterBlock(para, 1);
    problem.AddParameterBlock(para + 3, 1);
    problem.AddParameterBlock(para + 4, 1);
    problem.SetParameterBlockConstant(para + 3);
    problem.SetParameterBlockConstant(para + 4);

    for (auto &pair : keyframes)
    {
        auto position = pair.second->t();
        if (pair.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position, GetRawPoint(pair.second->feature_navsat->time), pair.second->feature_navsat->cov);
            problem.AddResidualBlock(cost_function, NULL, para, para + 3, para + 4);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    problem.SetParameterBlockVariable(para + 3);
    problem.SetParameterBlockVariable(para + 4);
    ceres::Solve(options, &problem, &summary);

    extrinsic = rpyxyz2se3(para);
    initialized = true;
}

void Navsat::Optimize(const Section &section)
{
    auto A = Map::Instance().GetKeyFrame(section.A);
    auto B = Map::Instance().GetKeyFrame(section.B);
    auto C = Map::Instance().GetKeyFrame(section.C);
    // first, optimize A-B
    // do not use all keyframes in BC, too much frames is not good
    if (A == B)
    {
        OptimizeRX(B, section.C, section.C, 0b111000);
    }
    else
    {
        OptimizeRX(B, section.C, section.C, 0b000000);
        // optimize A-B
        OptimizeAB(A, B, section.relative_B);
    }
    // second, optimize B-C
    Frames BC = Map::Instance().GetKeyFrames(section.B + epsilon, section.C - epsilon);
    for (auto &pair : BC)
    {
        auto frame = pair.second;
        OptimizeRX(frame, frame->time + epsilon, section.C, 0b110111);
    }
}

inline double navsat_distance(Frame::Ptr frame)
{
    Vector3d d = frame->t() - Navsat::Get()->GetFixPoint(frame);
    d.z() = 0;
    return d.norm();
}

void Navsat::QuickFix(double start, double end)
{
    if (PoseGraph::Instance().turning ||
        frames_distance(PoseGraph::Instance().current_section.B, end) < min_distance_fix_)
        return;
    auto A = Map::Instance().GetKeyFrame(start);
    auto B = Map::Instance().GetKeyFrame(PoseGraph::Instance().current_section.B);
    auto C = Map::Instance().GetKeyFrame(end);
    // first, optimize A-B
    // do not use all keyframes in BC, too much frames is not good
    OptimizeRX(B, C->time, C->time, A == B ? 0b111010 : 0b000000);
    // second, optimize B-C
    Frames BC = Map::Instance().GetKeyFrames(B->time + epsilon, C->time - epsilon);
    for (auto &pair : BC)
    {
        auto frame = pair.second;
        OptimizeRX(frame, frame->time + epsilon, C->time, 0b110111);
    }
    // if the orientation is wrong, add section
    // SE3d navsat_pose;
    // if (frames_distance(PoseGraph::Instance().current_section.B, end) > 200 &&
    //     EstimatePose(end, navsat_pose))
    // {
    //     Vector3d v1 = navsat_pose.so3() * Vector3d::UnitX();
    //     Vector3d v2 = C->pose.so3() * Vector3d::UnitX();
    //     if (vectors_degree_angle(v1, v2) > 5)
    //     {
    //         PoseGraph::Instance().AddSection(end);
    //     }
    // }
}

// mode: zyxrpy
void Navsat::OptimizeRX(Frame::Ptr frame, double end, double forward, unsigned char mode)
{
    // rotation's optimization need longer path
    if ((mode & 0b000111) != 0b000111 &&
        frames_distance(frame->time, end) < min_distance_fix_)
        return;
    SE3d old_pose = frame->pose;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, end);
    double para[6] = {0, 0, 0, 0, 0, 0};
    //NOTE: the real order of rpy is y p r
    problem.AddParameterBlock(para + 5, 1); //z
    problem.AddParameterBlock(para + 4, 1); //y
    problem.AddParameterBlock(para + 3, 1); //x
    problem.AddParameterBlock(para + 2, 1); //r
    problem.AddParameterBlock(para + 1, 1); //p
    problem.AddParameterBlock(para + 0, 1); //y
    for (int i = 0; i < 6; i++)
    {
        if (mode & (1 << i))
            problem.SetParameterBlockConstant(para + i);
    }
    // if para includes roll, ensure that vehicle can not roll over
    if (!problem.IsParameterBlockConstant(para + 2))
    {
        if (frames_distance(frame->time, end) > min_distance_fix_)
        {
            Vector3d y(0, 0, 0);
            for (auto &pair : active_kfs)
            {
                auto origin = pair.second->pose;
                y += frame->pose.inverse().so3() * origin.so3() * Vector3d::UnitY();
            }
            ceres::CostFunction *cost_function = NavsatRError::Create(y, frame->pose);
            problem.AddResidualBlock(cost_function, NULL, para + 2);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        }
        problem.SetParameterBlockConstant(para + 2);
    }
    // if distance is too small, dont optimize pitch
    if (!problem.IsParameterBlockConstant(para + 1) &&
        frames_distance(frame->time, end) < min_distance_fix_)
    {
        problem.SetParameterBlockConstant(para + 1);
    }

    for (auto &pair : active_kfs)
    {
        auto origin = pair.second->pose;
        if (pair.second->feature_navsat)
        {
            Vector3d point = GetFixPoint(pair.second);
            Vector3d cov = pair.second->feature_navsat->cov;
            ceres::CostFunction *cost_function = NavsatRXError::Create(point, frame->pose.inverse() * origin.translation(), frame->pose, cov);
            problem.AddResidualBlock(cost_function, loss_function, para, para + 1, para + 2, para + 3, para + 4, para + 5);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    frame->pose = frame->pose * rpyxyz2se3(para);
    SE3d new_pose = frame->pose;
    SE3d transform = new_pose * old_pose.inverse();
    PoseGraph::Instance().ForwardUpdate(transform, Map::Instance().GetKeyFrames(frame->time + epsilon, forward));
}

void Navsat::OptimizeAB(Frame::Ptr A, Frame::Ptr B, SE3d relative_B)
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    Frames AB = Map::Instance().GetKeyFrames(A->time + epsilon, B->time - epsilon);
    problem.AddParameterBlock(A->pose.data(), SE3d::num_parameters, local_parameterization);
    problem.AddParameterBlock(B->pose.data(), SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(A->pose.data());
    problem.SetParameterBlockConstant(B->pose.data());
    Frame::Ptr last_frame = A;
    for (auto &pair : AB)
    {
        auto frame = pair.second;
        // navsat point
        double a = (pair.first - A->time) / (B->time - A->time);
        Vector3d navsat_point = GetFixPoint(frame);
        navsat_point.z() = a * B->pose.translation().z() + (1 - a) * A->pose.translation().z();
        double *para_kf = frame->pose.data();
        double *para_last_kf = last_frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        ceres::CostFunction *cost_function1 = PoseGraphError::Create(last_frame->pose, frame->pose, 1, 20);
        problem.AddResidualBlock(cost_function1, NULL, para_last_kf, para_kf);
        ceres::CostFunction *cost_function2 = TError::Create(navsat_point);
        problem.AddResidualBlock(cost_function2, loss_function, para_kf);
        last_frame = frame;
    }
    ceres::CostFunction *cost_function1 = PoseGraphError::Create(relative_B, 10, 20);
    problem.AddResidualBlock(cost_function1, NULL, last_frame->pose.data(), B->pose.data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

} // namespace lvio_fusion
