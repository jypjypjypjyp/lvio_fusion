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
        Vector3d dirG = Vector3d::Zero();
        // bool isfirst=true;
        Vector3d velocity;
        bool firstframe = true;
        int i = 1;
        for (std::vector<Frame::Ptr>::iterator iter_keyframe = keyframes.begin() + 1; iter_keyframe != keyframes.end(); iter_keyframe++)
        {
            if ((*iter_keyframe)->preintegration == nullptr)
            {
                return false;
            }
            if (!(*iter_keyframe)->last_keyframe)
            {
                continue;
            }
            i++;
            dirG += (*iter_keyframe)->last_keyframe->GetImuRotation() * (*iter_keyframe)->preintegration->GetUpdatedDeltaVelocity();
            velocity = ((*iter_keyframe)->GetImuPosition() - (*(iter_keyframe))->last_keyframe->GetImuPosition()) / ((*iter_keyframe)->preintegration->sum_dt);
            (*iter_keyframe)->SetVelocity(velocity);
            (*iter_keyframe)->last_keyframe->SetVelocity(velocity);
        }
        dirG = dirG / dirG.norm();

        Vector3d gI(0.0, 0.0, 1.0); //沿-z的归一化的重力数值
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v * ang / nv;
        if (bimu)
        {
            Rwg_ = Imu::Get()->Rwg;
        }
        else
        {
            Rwg_ = ExpSO3(vzg);
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

bool Initializer::Initialize(Frames keyframes, double priorA, double priorG)
{
    double minTime = 20.0; // 初始化需要的最小时间间隔
    // 按时间顺序收集初始化imu使用的KF
    std::list<Frame::Ptr> KeyFrames_list;
    Frames::reverse_iterator iter;
    for (iter = keyframes.rbegin(); iter != keyframes.rend(); iter++)
    {
        KeyFrames_list.push_front(iter->second);
    }
    std::vector<Frame::Ptr> Key_frames(KeyFrames_list.begin(), KeyFrames_list.end());

    const int N = Key_frames.size(); // 待处理的关键帧数目

    // 估计KF速度和重力方向
    if (!EstimateVelAndRwg(Key_frames))
    {
        return false;
    }
    bool isOptRwg = true; //reinit||!bimu;
    bool isOK;
    if (priorA == 0)
    {
        isOK = imu::InertialOptimization(keyframes, Rwg_, 1e2, 1e4, isOptRwg);
    }
    else
    {
        isOK = imu::InertialOptimization(keyframes, Rwg_, priorG, priorA, isOptRwg);
    }
    if (!isOK)
    {
        return false;
    }
    Vector3d dirG;
    dirG << 0, 0, Imu::Get()->G;
    dirG = Rwg_ * dirG;
    dirG = dirG / dirG.norm();
    if (!(dirG[0] == 0 && dirG[1] == 0 && dirG[2] == 1))
    {
        Vector3d gI(0.0, 0.0, 1.0); //沿-z的归一化的重力数值
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v * ang / nv;
        Rwg_ = ExpSO3(vzg);
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

    for (int i = 0; i < N; i++)
    {
        Frame::Ptr pKF2 = Key_frames[i];
        pKF2->bImu = true;
    }

    if (priorA != 0)
    {
        imu::FullInertialBA(keyframes, priorG, priorA);
    }

    bimu = true;
    Imu::Get()->initialized = true;
    reinit = false;
    return true;
}

} // namespace lvio_fusion
