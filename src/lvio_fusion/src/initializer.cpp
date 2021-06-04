#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void Initializer::EstimateVelAndRwg(Frames frames)
{
    if (!Imu::Get()->initialized)
    {
        Vector3d twg = Vector3d::Zero();
        Vector3d Vw;
        for (auto &pair : frames)
        {
            auto frame = pair.second;
            twg += frame->last_keyframe->R() * frame->preintegration->GetUpdatedDeltaVelocity();
            Vw = (frame->t() - frame->last_keyframe->t()) / frame->preintegration->sum_dt;
            frame->SetVelocity(Vw);
            frame->SetBias(Bias());
        }
        if (step != 4)
        {
            Rwg_ = get_R_from_vector(twg);
        }
    }
}

// make sure than every frame has last_frame and preintegrate
bool Initializer::Initialize(Frames frames, double prior_a, double prior_g)
{
    // estimate velocity and gravity direction
    EstimateVelAndRwg(frames);

    // imu optimization (don't change gravity when step == 4)
    if (step != 4)
    {
        if (!imu::InertialOptimization(frames, Rwg_, prior_a, prior_g))
            return false;
        Rwg_ = get_R_from_vector(Rwg_ * Vector3d::UnitZ());
        Map::Instance().ApplyGravityRotation(Rwg_.inverse());
    }

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
            step = step != 4 ? 1 : 4;
            Imu::Get()->initialized = false;
            LOG(INFO) << "Initializer Failed";
        }
    }
}

} // namespace lvio_fusion
