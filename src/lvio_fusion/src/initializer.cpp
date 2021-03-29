#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/imu/tools.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

bool Initializer::EstimateVelAndRwg(std::vector<Frame::Ptr> keyframes)
{
    if (!Imu::Get()->initialized)
    {
        Vector3d ori_G = Vector3d::Zero();
        Vector3d velocity;
        for (auto iter = keyframes.begin() + 1; iter != keyframes.end(); iter++)
        {
            if ((*iter)->preintegration == nullptr)
                return false;
            if (!(*iter)->last_keyframe)
                continue;

            ori_G += (*iter)->last_keyframe->GetImuRotation() * (*iter)->preintegration->GetUpdatedDeltaVelocity();
            velocity = ((*iter)->GetImuPosition() - (*(iter))->last_keyframe->GetImuPosition()) / ((*iter)->preintegration->sum_dt);
            (*iter)->SetVelocity(velocity);
            (*iter)->last_keyframe->SetVelocity(velocity);
        }
        ori_G = ori_G / ori_G.norm();

        Vector3d gI(0.0, 0.0, 1.0);
        Vector3d v = gI.cross(ori_G);
        const double nv = v.norm();
        const double cosg = gI.dot(ori_G);
        const double ang = acos(cosg);
        Vector3d vzg = v * ang / nv;
        if (first_init)
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

bool Initializer::Initialize(Frames frames, double priorA, double priorG)
{
    double min_dt = 20.0;
    std::list<Frame::Ptr> kfs_list;
    for (Frames::reverse_iterator iter = frames.rbegin(); iter != frames.rend(); iter++)
    {
        kfs_list.push_front(iter->second);
    }
    std::vector<Frame::Ptr> kfs(kfs_list.begin(), kfs_list.end());

    // estimate velocity and gravity direction
    if (!EstimateVelAndRwg(kfs))
        return false;
    bool ok;
    if (priorA == 0)
    {
        ok = imu::InertialOptimization(frames, Rwg_, 1e2, 1e4);
    }
    else
    {
        ok = imu::InertialOptimization(frames, Rwg_, priorG, priorA);
    }
    if (!ok)
        return false;

    Vector3d dirG;
    dirG << 0, 0, Imu::Get()->G;
    dirG = Rwg_ * dirG;
    dirG = dirG / dirG.norm();
    if (!(dirG[0] == 0 && dirG[1] == 0 && dirG[2] == 1))
    {
        Vector3d gI(0.0, 0.0, 1.0);
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        const double cosg = gI.dot(dirG);
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

    for (int i = 0; i < kfs.size(); i++)
    {
        Frame::Ptr pKF2 = kfs[i];
        pKF2->is_imu_good = true;
    }

    if (priorA != 0)
    {
        imu::FullInertialBA(frames, priorG, priorA);
    }

    first_init = true;
    Imu::Get()->initialized = true;
    need_reinit = false;
    return true;
}

} // namespace lvio_fusion
