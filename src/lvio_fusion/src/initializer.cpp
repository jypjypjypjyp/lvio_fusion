#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{
void InertialOptimization(Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, double priorG = 1, double priorA = 1e9)
{
    Frames key_frames = Map::Instance().GetAllKeyFrames();
    ceres::Problem problem;

    //先验BIAS约束
    auto para_gyroBias=key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);
    ceres::CostFunction *cost_function = PriorGyroError::Create(Vector3d::Zero(),priorG);
    problem.AddResidualBlock(cost_function,NULL,para_gyroBias);

    auto para_accBias=key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);
    cost_function = PriorAccError::Create(Vector3d::Zero(),priorA);
    problem.AddResidualBlock(cost_function,NULL,para_accBias);
    
    //优化重力、BIAS和速度的边
    Vector3d rwg=Rwg.eulerAngles(0,1,2);
    auto para_rwg=rwg.data();
    problem.AddParameterBlock(para_rwg, 3);
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    double *para_scale=&scale;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if (!current_frame->bImu||!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_v, 3);   

        if (last_frame && current_frame->bImu&&last_frame->last_keyframe)
        {
            auto para_v_last = last_frame->Vw.data();
            cost_function = InertialGSError::Create(current_frame->preintegration,current_frame->pose,last_frame->pose);
            problem.AddResidualBlock(cost_function, NULL,para_v_last,  para_v,para_gyroBias,para_accBias,para_rwg);
        }
        last_frame = current_frame;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::this_thread::sleep_for(std::chrono::seconds(3));

    //数据恢复
    Eigen::AngleAxisd rollAngle(AngleAxisd(rwg(0),Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(AngleAxisd(rwg(1),Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(AngleAxisd(rwg(2),Vector3d::UnitZ()));
    Rwg= yawAngle*pitchAngle*rollAngle;

    Bias bias_(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
    bg << para_gyroBias[0],para_gyroBias[1],para_gyroBias[2];
    ba <<para_accBias[0],para_accBias[1],para_accBias[2];
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        Vector3d dbg=current_frame->GetGyroBias()-bg;
        if(dbg.norm() >0.01)
        {
            current_frame->SetNewBias(bias_);
            if (current_frame->bImu)
                current_frame->preintegration->Reintegrate();
        }
       else
        {
            current_frame->SetNewBias(bias_);
        }     
    }
}

void FullInertialBA(Matrix3d Rwg, double priorG = 1, double priorA=1e9)
{
    Frames KFs=Map::Instance().GetAllKeyFrames();
    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));

    Frame::Ptr pIncKF=KFs.begin()->second;
    double* para_gyroBias;
    double* para_accBias;
    para_gyroBias=pIncKF->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_accBias, 3);   
    para_accBias=pIncKF->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);   

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    int i=KFs.size();
    for (auto kf_pair : KFs)
    {
        i--;
        current_frame = kf_pair.second;
        if (!current_frame->bImu||!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_kf = current_frame->pose.data();
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->bImu&&last_frame->last_keyframe)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            ceres::CostFunction *cost_function =InertialError::Create(current_frame->preintegration,Rwg);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last,  para_gyroBias,para_accBias, para_kf, para_v);
        }
        last_frame = current_frame;
    }
    //epa  para_accBias
    ceres::CostFunction *cost_function = PriorAccError::Create(Vector3d::Zero(),priorA);
    problem.AddResidualBlock(cost_function, NULL, para_accBias);
    //epg para_gyroBias
    cost_function = PriorGyroError::Create(Vector3d::Zero(),priorG);
    problem.AddResidualBlock(cost_function, NULL, para_gyroBias);
    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //数据恢复
    for(Frames::iterator iter = KFs.begin(); iter != KFs.end(); iter++)
    {
        current_frame=iter->second;
        if(current_frame->bImu&&current_frame->last_keyframe)
        {
            Bias bias(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
            current_frame->SetNewBias(bias);
        }
    }
}

void  Initializer::InitializeIMU(bool bFIBA)
{
    double priorA=1e6;
    double priorG=1e2;
    double minTime=1.0;  // 初始化需要的最小时间间隔
    // 按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > KeyFrames_list;
    Frames keyframes=Map::Instance().GetAllKeyFrames();
    Frames::reverse_iterator   iter;
    for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++)
    {
        KeyFrames_list.push_front(iter->second);
    }
    std::vector< Frame::Ptr > Key_frames(KeyFrames_list.begin(),KeyFrames_list.end());
    unsigned long last_initialized_id=Key_frames.back()->id;
    // imu计算初始时间
    FirstTs=Key_frames.front()->time;
    if(Map::Instance().current_frame->time-FirstTs<minTime)
        return;

    const int N = Key_frames.size();  // 待处理的关键帧数目

    // 估计KF速度和重力方向
    if (!initialized)
    {
        Vector3d dirG = Vector3d::Zero();
        for(std::vector<Frame::Ptr>::iterator iter_keyframe = Key_frames.begin()+1; iter_keyframe!=Key_frames.end(); iter_keyframe++) 
        {
            if (!(*iter_keyframe)->preintegration->isPreintegrated) 
            {
                continue;                
            }
            if(!(*iter_keyframe)->last_keyframe)
            {
                continue;
            }
            dirG -= (*iter_keyframe)->last_keyframe->GetImuRotation()*(*iter_keyframe)->preintegration->GetUpdatedDeltaVelocity();
            if(!(*iter_keyframe)->last_keyframe->last_keyframe)
            {
                continue;
            }
            Vector3d velocity = ((*iter_keyframe)->GetImuPosition() - (*(iter_keyframe))->last_keyframe->last_keyframe->GetImuPosition())/((*iter_keyframe)->last_keyframe->preintegration->dT+ (*(iter_keyframe))->last_keyframe->last_keyframe->preintegration->dT);
            (*iter_keyframe)->last_keyframe->SetVelocity(velocity);
        }
        dirG = dirG/dirG.norm();
        Vector3d  gI(0.0, 0.0, -1.0);//沿-z的归一化的重力数值
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v*ang/nv;
        Rwg = ExpSO3(vzg);

        Frame::Ptr current_frame=(*(--Key_frames.end()));
        Vector3d Gz ;
        Gz << 0, 0, -9.81007;
        Gz =Rwg*Gz;
        double t12=current_frame->preintegration->dT;
         Vector3d twb1=current_frame->last_keyframe->GetImuPosition();
        Vector3d twb2=current_frame->GetImuPosition();
        Vector3d Vwb2=current_frame->last_keyframe->Vw+t12*Gz+current_frame->preintegration->GetDeltaRotation(current_frame->GetImuBias())*current_frame->preintegration->GetDeltaVelocity(current_frame->GetImuBias());
        current_frame->SetVelocity( Vwb2);
    }
    else
    {
        bg = Map::Instance().current_frame->GetGyroBias();
        ba =Map::Instance().current_frame->GetAccBias();
    }
 
    Scale=1.0;

    InertialOptimization(Rwg, Scale, bg, ba, priorG, priorA);
    if (Scale<1e-1)
    {
        return;
    }
    frontend_.lock()->UpdateFrameIMU(Scale,Key_frames[0]->GetImuBias(),frontend_.lock()->current_key_frame);
    //更新关键帧中imu状态
    for(int i=0;i<N;i++)
    {
        Frame::Ptr pKF2 = Key_frames[i];
        pKF2->bImu = true;
    }
    // 进行完全惯性优化
    FullInertialBA(Rwg, priorG, priorA);
    frontend_.lock()->UpdateFrameIMU(1.0,Key_frames[last_initialized_id-1]->GetImuBias(),frontend_.lock()->current_key_frame);
    frontend_.lock()->current_key_frame->bImu = true;
    bimu=true;
    return;
}

} // namespace lvio_fusioni