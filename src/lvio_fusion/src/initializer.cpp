#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

bool Initializer::EstimateVelAndRwg(Frames frames)
{
    if (!Imu::Get()->initialized)
    {
        Vector3d Ig = Vector3d::Zero();
        Vector3d velocity;
        for (auto &pair : frames)
        {
            auto frame = pair.second;
            if (!frame->preintegration || !frame->last_keyframe)
                return false;

            Ig += frame->last_keyframe->GetRotation() * frame->preintegration->GetUpdatedDeltaVelocity();
            velocity = (frame->GetPosition() - frame->last_keyframe->GetPosition()) / frame->preintegration->sum_dt;
            frame->SetVelocity(velocity);
            frame->last_keyframe->SetVelocity(velocity);
        }
        Ig = Ig / Ig.norm();

        Vector3d Iz(0.0, 0.0, 1.0);
        Vector3d v = Iz.cross(Ig);
        Vector3d vzg = v * acos(Iz.dot(Ig)) / v.norm();
        if (finished_first_init)
        {
            Rwg_ = Imu::Get()->Rwg;
        }
        else
        {
            Rwg_ = exp_so3(vzg);
        }
        Vector3d g;
        g << 0, 0, Imu::Get()->G;
        g = Rwg_ * g;
        LOG(INFO) << "Gravity Vector: " << (g).transpose();
    }
    else
    {
        Rwg_ = Imu::Get()->Rwg; //* Matrix3d::Identity();
    }
    return true;
}

bool Initializer::Initialize(Frames frames, double prior_a, double prior_g)
{
    // estimate velocity and gravity direction
    if (!EstimateVelAndRwg(frames))
        return false;

    // imu optimization
    bool success;
    if (prior_a == 0)
    {
        success = imu::InertialOptimization(frames, Rwg_, 1e2, 1e4);
    }
    else
    {
        success = imu::InertialOptimization(frames, Rwg_, prior_g, prior_a);
    }
    if (!success)
        return false;

    Vector3d twg;
    twg << 0, 0, Imu::Get()->G;
    twg = Rwg_ * twg;
    twg = twg / twg.norm();
    if (!(twg[0] == 0 && twg[1] == 0 && twg[2] == 1))
    {
        Vector3d gI(0.0, 0.0, 1.0);
        Vector3d v = gI.cross(twg);
        const double nv = v.norm();
        const double cosg = gI.dot(twg);
        const double ang = acos(cosg);
        Vector3d vzg = v * ang / nv;
        Rwg_ = exp_so3(vzg);
    }
    else
    {
        Rwg_ = Matrix3d::Identity();
    }
    Vector3d g2;
    g2 << 0, 0, Imu::Get()->G;
    g2 = Rwg_ * g2;
    LOG(INFO) << "Gravity Vector again: " << (g2).transpose();
    Imu::Get()->Rwg = Rwg_;

    for (auto &pair : frames)
    {
        pair.second->good_imu = true;
    }

    // imu optimization with visual
    if (prior_a != 0)
    {
        imu::FullInertialBA(frames, prior_g, prior_a);
    }

    finished_first_init = true;
    Imu::Get()->initialized = true;
    return true;
}

} // namespace lvio_fusion
