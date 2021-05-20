#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

bool Initializer::EstimateVelAndRwg(Frames frames)
{
    if (!Imu::Get()->initialized)
    {
        Vector3d twg = Vector3d::Zero();
        Vector3d Vw;
        for (auto &pair : frames)
        {
            auto frame = pair.second;
            twg += frame->last_keyframe->GetRotation() * frame->preintegration->GetUpdatedDeltaVelocity();
            Vw = (frame->GetPosition() - frame->last_keyframe->GetPosition()) / frame->preintegration->sum_dt;
            frame->SetVelocity(Vw);
        }
        Rwg_ = get_R_from_vector(twg);
        Vector3d g(0, 0, Imu::Get()->G);
        g = Rwg_ * g;
        LOG(INFO) << "Gravity Vector: " << g.transpose();
    }
    else
    {
        Rwg_ = Imu::Get()->Rwg;
    }
    return true;
}

// make sure than every frame has last_frame and preintegrate
bool Initializer::Initialize(Frames frames, double prior_a, double prior_g)
{
    // estimate velocity and gravity direction
    if (!EstimateVelAndRwg(frames))
        return false;

    // imu optimization
    if (!imu::InertialOptimization(frames, Rwg_, prior_a, prior_g))
        return false;

    Rwg_ = get_R_from_vector(Rwg_ * Vector3d::UnitZ());
    Vector3d g2(0, 0, Imu::Get()->G);
    g2 = Rwg_ * g2;
    LOG(INFO) << "Gravity Vector again: " << g2.transpose();
    Imu::Get()->Rwg = Rwg_;

    for (auto &pair : frames)
    {
        pair.second->good_imu = true;
    }

    // imu optimization with visual
    imu::FullBA(frames, prior_a, prior_g);
    Imu::Get()->initialized = true;
    return true;
}

// 3-step initialization
void Initializer::Initialize(double init_time, double end_time)
{
    static double last_init_time = 0;
    bool need_init = false;
    double prior_a = 1e4, prior_g = 1e2;
    if (Imu::Get()->initialized)
    {
        double dt = last_init_time ? end_time - last_init_time : 0;
        if (dt > 5 && step == 2)
        {
            need_init = true;
            step = 3;
        }
        else if (dt > 5 && step == 3)
        {
            need_init = true;
            step = 4;
        }
    }
    else
    {
        if (step == 4)
        {
            need_init = true;
            step = 4;
        }
        else
        {
            need_init = true;
            step = 2;
        }
    }

    Frames frames_init;
    SE3d old_pose, new_pose;
    if (need_init)
    {
        need_init = false;
        frames_init = Map::Instance().GetKeyFrames(init_time, end_time);
        if (frames_init.size() >= num_frames_init &&
            frames_init.begin()->second->preintegration)
        {
            old_pose = (--frames_init.end())->second->pose;
            if (!Imu::Get()->initialized)
            {
                last_init_time = (--frames_init.end())->second->time;
            }
            need_init = true;
        }
    }
    if (need_init)
    {
        LOG(INFO) << "Initializer Start";
        if (Initialize(frames_init, prior_a, prior_g))
        {
            new_pose = (--frames_init.end())->second->pose;
            SE3d transform = new_pose * old_pose.inverse();
            PoseGraph::Instance().ForwardUpdate(transform, end_time, false);
            for (auto &pair : frames_init)
            {
                if (pair.second->preintegration)
                    pair.second->good_imu = true;
            }
            LOG(INFO) << "Initializer Finished";
        }
        else
        {
            step = 1;
            Imu::Get()->initialized = false;
            LOG(INFO) << "Initializer Failed";
        }
    }
}

} // namespace lvio_fusion
