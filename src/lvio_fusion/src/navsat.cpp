#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/pose_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void Navsat::AddPoint(double time, double x, double y, double z)
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
        pair.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(pair.first));
        // check
        // pair.second->feature_navsat->trust = (p.z() > -3 && p.z() < 3);
        pair.second->feature_navsat->trust = true;
        finished = pair.first + epsilon;
    }
    if (!initialized && !Map::Instance().keyframes.empty() && frames_distance(0, -1) > 40)
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
    auto iter = raw.lower_bound(time);
    if (iter == raw.end())
        iter--;
    return extrinsic * iter->second;
}

SE3d Navsat::GetAroundPose(double time)
{
    auto iter1 = raw.lower_bound(time);
    auto iter2 = iter1--;
    if (iter2 == raw.begin())
    {
        iter1--;
        iter2--;
    }
    Vector3d p1 = extrinsic * iter1->second;
    Vector3d p2 = extrinsic * iter2->second;
    while (iter1 != raw.begin())
    {
        if ((p1 - p2).norm() > 2)
        {
            return get_pose_from_two_points(p1, p2);
        }
        else
        {
            iter1--;
            p1 = extrinsic * iter1->second;
        }
    }
    return SE3d();
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
        auto position = pair.second->GetPosition();
        if (pair.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position, GetRawPoint(pair.second->feature_navsat->time));
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
    // optimize B - C
    // first, optimize B's position
    OptimizeRX(B, std::min(B->time + 3, section.C), section.C, 0b000000);
    // second, optimize B's rotation
    OptimizeRX(B, section.C, section.C, 0b111000);
    // third, optimize others
    Frames BC = Map::Instance().GetKeyFrames(section.B + epsilon, section.C - epsilon);
    for (auto &pair : BC)
    {
        auto frame = pair.second;
        OptimizeRX(frame, std::min(frame->time + 3, section.C), section.C, 0b110111);
    }
    // optimize A - B
    // first, optimize A's rotation
    OptimizePlane(A, section.B - epsilon, section.B - epsilon, A->GetPosition(), B->GetPosition(), C->GetPosition());
    // second, optimize others
    Frames AB = Map::Instance().GetKeyFrames(section.A + epsilon, section.B - epsilon);
    for (auto &pair : AB)
    {
        auto frame = pair.second;
        OptimizeRX(frame, section.B - epsilon, section.B - epsilon, 0b110011);
    }
}

double Navsat::QuickFix(double current_time, double end_time)
{
    // Frame::Ptr current_frame = Map::Instance().GetKeyFrame(current_time);
    // Frame::Ptr finished_frame = Map::Instance().GetKeyFrame(finished);
    // Vector3d t = current_frame->GetPosition() - GetFixPoint(current_frame);
    // current_time = current_frame->time;
    // double B = PoseGraph::Instance().current_section.B;
    // if (t.norm() > 2 &&
    //     !PoseGraph::Instance().turning &&
    //     frames_distance(current_time, B) > 60)
    // {
    //     // optimize A - B
    //     OptimizeZ(finished_frame, end_time);
    //     OptimizeRX(finished_frame, current_time, end_time, 8);
    //     Frames AB_kfs = Map::Instance().GetKeyFrames(finished + epsilon, B - epsilon);
    //     for (auto &pair : AB_kfs)
    //     {
    //         auto frame = pair.second;
    //         // OptimizeRX(frame, std::min(B - epsilon, frame->time + 1), end_time, 1 + 2 + 4);
    //         OptimizeRX(frame, std::min(frame->time + 3, B - epsilon), end_time, 8);
    //     }
    //     // optimize B - C
    //     OptimizeRX(finished_frame, current_time, end_time, 8);
    //     Frames BC_kfs = Map::Instance().GetKeyFrames(B + epsilon, current_time - epsilon);
    //     for (auto &pair : BC_kfs)
    //     {
    //         auto frame = pair.second;
    //         OptimizeRX(frame, std::min(frame->time + 1, end_time), end_time, 1 + 2 + 4);
    //     }

    //     t = current_frame->GetPosition() - GetFixPoint(current_frame);
    //     t.z() = 0;
    //     if (t.norm() > 2)
    //     {
    //         PoseGraph::Instance().AddSection(current_time);
    //     }
    //     return finished;
    // }
    // return 0;
}

// mode: zyxrpy
void Navsat::OptimizeRX(Frame::Ptr frame, double end, double forward_time, unsigned char mode)
{
    if (!(mode & (1 << 1) && mode & (1 << 2)) && frames_distance(frame->time, end) < 10)
        return;
    SE3d old_pose = frame->pose;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, end);
    double para[6] = {0, 0, 0, 0, 0, 0};
    //NOTE: the real order of rpy is y p r
    problem.AddParameterBlock(para + 5, 1); //z
    problem.AddParameterBlock(para + 4, 1); //y
    problem.AddParameterBlock(para + 3, 1); //x
    problem.AddParameterBlock(para + 2, 1); //r
    problem.AddParameterBlock(para + 1, 1); //p
    problem.AddParameterBlock(para + 0, 1); //y
    for (int i = 0; i < SE3d::DoF; i++)
    {
        if (mode & (1 << i))
            problem.SetParameterBlockConstant(para + i);
    }

    Vector3d dp = frame->GetPosition() - GetFixPoint(frame);
    for (auto &pair : active_kfs)
    {
        auto origin = pair.second->pose;
        if (pair.second->feature_navsat)
        {
            Vector3d point = GetFixPoint(pair.second);
            if (!pair.second->feature_navsat->trust)
            {
                point.z() = frame->GetPosition().z();
            }
            ceres::CostFunction *cost_function = NavsatRXError::Create(point, frame->pose.inverse() * origin.translation(), frame->pose);
            problem.AddResidualBlock(cost_function, loss_function, para, para + 1, para + 2, para + 3, para + 4, para + 5);
        }
    }

    // if para includes rotation, ensure that vehicle can not roll over
    if ((mode & 0b000111) != 0b000111)
    {
        for (auto &pair : active_kfs)
        {
            auto origin = pair.second->pose;
            ceres::CostFunction *cost_function = NavsatRError::Create(origin, frame->pose);
            problem.AddResidualBlock(cost_function, NULL, para, para + 1, para + 2);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    frame->pose = frame->pose * rpyxyz2se3(para);
    SE3d new_pose = frame->pose;
    SE3d transform = new_pose * old_pose.inverse();
    PoseGraph::Instance().ForwardUpdate(transform, Map::Instance().GetKeyFrames(frame->time + epsilon, forward_time));
}

void Navsat::OptimizePlane(Frame::Ptr frame, double end, double forward_time, Vector3d A, Vector3d B, Vector3d C)
{
    SE3d old_pose = frame->pose;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, end);
    double para[6] = {0, 0, 0, 0, 0, 0};
    //NOTE: the real order of rpy is y p r
    problem.AddParameterBlock(para + 2, 1); //r
    problem.AddParameterBlock(para + 1, 1); //p
    problem.AddParameterBlock(para + 0, 1); //y
    problem.SetParameterBlockConstant(para + 2);

    for (auto &pair : active_kfs)
    {
        auto origin = pair.second->pose;
        if (pair.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatPlaneError::Create(frame->pose.inverse() * origin.translation(), A, B, C, frame->pose);
            problem.AddResidualBlock(cost_function, loss_function, para, para + 1, para + 2);
        }
    }

    for (auto &pair : active_kfs)
    {
        auto origin = pair.second->pose;
        ceres::CostFunction *cost_function = NavsatRError::Create(origin, frame->pose);
        problem.AddResidualBlock(cost_function, NULL, para, para + 1, para + 2);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    frame->pose = frame->pose * rpyxyz2se3(para);
    SE3d new_pose = frame->pose;
    SE3d transform = new_pose * old_pose.inverse();
    PoseGraph::Instance().ForwardUpdate(transform, Map::Instance().GetKeyFrames(frame->time + epsilon, forward_time));
}

} // namespace lvio_fusion
