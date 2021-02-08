#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{

void InertialOptimization(Frames &key_frames, Eigen::Matrix3d &Rwg,double priorG, double priorA)   
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function ;
    //先验BIAS约束
    auto para_gyroBias=key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);

    auto para_accBias=key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);

    //优化重力、BIAS和速度的边
    Quaterniond rwg(Rwg);
    SO3d RwgSO3(rwg);
    auto para_rwg=RwgSO3.data();
    
    ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(para_rwg, SO3d::num_parameters,local_parameterization);
    //problem.SetParameterBlockConstant(para_rwg);
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if (!current_frame->last_keyframe||current_frame->preintegration==nullptr)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_v, 3);   

        if (last_frame)
        {
            auto para_v_last = last_frame->Vw.data();
            cost_function = ImuErrorG::Create(current_frame->preintegration,current_frame->pose,last_frame->pose,priorA,priorG);
            problem.AddResidualBlock(cost_function, NULL,para_v_last,para_accBias,para_gyroBias,para_v,para_rwg);
        }
        last_frame = current_frame;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
      options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_solver_time_in_seconds = 0.1;
   //  options.max_num_iterations =4;
  
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
     LOG(INFO)<<summary.BriefReport();
   // std::this_thread::sleep_for(std::chrono::seconds(3));

    //数据恢复
    Quaterniond rwg2(RwgSO3.data()[3],RwgSO3.data()[0],RwgSO3.data()[1],RwgSO3.data()[2]);
        Rwg=rwg2.toRotationMatrix();
    Bias bias_(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
    Vector3d bg;
    bg<< para_gyroBias[0],para_gyroBias[1],para_gyroBias[2];
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        Vector3d dbg=current_frame->GetGyroBias()-bg;
        if(dbg.norm() >0.01)
        {
            current_frame->SetNewBias(bias_);
            current_frame->preintegration->Repropagate(bias_.linearized_ba,bias_.linearized_bg);
        }
       else
        {
            current_frame->SetNewBias(bias_);
        }     
            Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
        // LOG(INFO)<<"InertialOptimization  "<<current_frame->time-1.40364e+09+8.60223e+07<<"   Vwb1  "<<current_frame->Vw.transpose()/*tcb*/<<"  Pose  "<<current_frame->pose.translation().transpose()/*tcb*/;//<<"\nR: \n"<<tcb.inverse()*current_frame->pose.rotationMatrix();
    }
    // LOG(INFO)<<"BIAS   a "<<bias_.linearized_ba.transpose()<<" g "<<bias_.linearized_bg.transpose();
    
    return ;
}

void FullInertialBA(Frames &key_frames, double priorG=1e2, double priorA=1e6)
{
    ceres::Problem problem;
    ceres::CostFunction *cost_function ;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = key_frames.begin()->first;
    SE3d old_pose=key_frames.begin()->second->pose;
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

    Frame::Ptr pIncKF=key_frames.begin()->second;

    auto para_gyroBias=key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);

    auto para_accBias=key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    int i=0;
    for (auto kf_pair : key_frames)
    {
        current_frame = kf_pair.second;
        i++;
        if (!current_frame->bImu||!current_frame->last_keyframe||current_frame->preintegration==nullptr)
        {
            last_frame=current_frame;
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

            cost_function =ImuErrorInit::Create(current_frame->preintegration,priorA,priorG);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_accBias,para_gyroBias, para_kf, para_v);
        }
        last_frame = current_frame;
    }

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations=4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.BriefReport();
    //数据恢复
    Bias lastBias;

    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if(current_frame->bImu&&current_frame->last_keyframe)
        {
            Bias bias(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
            current_frame->SetNewBias(bias);
            lastBias=bias;
        }
    }
    return ;
}

void recoverData(Frames active_kfs,SE3d old_pose)
{
    SE3d new_pose=active_kfs.begin()->second->pose;
    Vector3d origin_P0=old_pose.translation();
    Vector3d origin_R0=R2ypr( old_pose.rotationMatrix());
  Vector3d origin_R00=R2ypr(new_pose.rotationMatrix());
 double y_diff = origin_R0.x() - origin_R00.x();
   Matrix3d rot_diff = ypr2R(Vector3d(y_diff, 0, 0));
   if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            rot_diff =old_pose.rotationMatrix() *new_pose.inverse().rotationMatrix();
        }
        for(auto kf_pair : active_kfs){
            auto frame = kf_pair.second;
            if(!frame->preintegration||!frame->last_keyframe||!frame->bImu){
                    continue;
            }
            frame->SetPose(rot_diff * frame->pose.rotationMatrix(),rot_diff * (frame->pose.translation()-new_pose.translation())+origin_P0);
            frame->SetVelocity(rot_diff*frame->Vw);
            frame->SetNewBias(frame->GetImuBias());
        }
}

void FullInertialBA2(Frames &key_frames)
{
    ceres::Problem problem;
      ceres::CostFunction *cost_function ;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = key_frames.begin()->first;
SE3d old_pose=key_frames.begin()->second->pose;
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

    Frame::Ptr pIncKF=key_frames.begin()->second;

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    int i=0;
    for (auto kf_pair : key_frames)
    {
        current_frame = kf_pair.second;
        i++;
        if (!current_frame->bImu||!current_frame->last_keyframe||current_frame->preintegration==nullptr)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_kf = current_frame->pose.data();
        auto para_v = current_frame->Vw.data();
        auto para_gyroBias=current_frame->ImuBias.linearized_bg.data();
        auto para_accBias=current_frame->ImuBias.linearized_ba.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        problem.AddParameterBlock(para_gyroBias, 3);
        problem.AddParameterBlock(para_accBias, 3);
        if (last_frame && last_frame->bImu)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            auto para_gyroBias_last=last_frame->ImuBias.linearized_bg.data();
            auto para_accBias_last=last_frame->ImuBias.linearized_ba.data();
            cost_function =ImuError::Create(current_frame->preintegration);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_accBias_last,para_gyroBias_last, para_kf, para_v,para_accBias,para_gyroBias);

        }
        last_frame = current_frame;
    }
    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations=4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.BriefReport();
    //数据恢复
    
    recoverData(key_frames,old_pose);
    Bias lastBias;
    return;
}

bool  Initializer::estimate_Vel_Rwg(std::vector< Frame::Ptr > Key_frames)
{
    
    if (!initialized)
    {
        Vector3d dirG = Vector3d::Zero();
       // bool isfirst=true;
        Vector3d velocity;
        bool firstframe=true;
        int i=1;
        for(std::vector<Frame::Ptr>::iterator iter_keyframe = Key_frames.begin()+1; iter_keyframe!=Key_frames.end(); iter_keyframe++) 
        {
            if((*iter_keyframe)->preintegration==nullptr)
            {
                return false;
            }   
            if(!(*iter_keyframe)->last_keyframe)
            {
                continue;
            }
            i++;
            dirG += (*iter_keyframe)->last_keyframe->GetImuRotation()*(*iter_keyframe)->preintegration->GetUpdatedDeltaVelocity();
            velocity= ((*iter_keyframe)->GetImuPosition() - (*(iter_keyframe))->last_keyframe->GetImuPosition())/((*iter_keyframe)->preintegration->sum_dt);
            (*iter_keyframe)->SetVelocity(velocity);
            (*iter_keyframe)->last_keyframe->SetVelocity(velocity);
                
        }
        dirG = dirG/dirG.norm();
  
        Vector3d  gI(0.0, 0.0, 1.0);//沿-z的归一化的重力数值
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v*ang/nv;
        if(bimu){
            Rwg=Matrix3d::Identity();
        }else{
            Rwg = ExpSO3(vzg);
        }
        Vector3d g;
         g<< 0, 0, 9.8007;
         g=Rwg*g;
       LOG(INFO)<<"INITG "<<(g).transpose();
    } 
    else
    {
        Rwg=Matrix3d::Identity();
    }
    return true;
}

void ApplyScaledRotation(const Matrix3d &R,Frames keyframes)
{
 SE3d pose0=keyframes.begin()->second->pose;
     Matrix3d new_pose=R*pose0.rotationMatrix();
    Vector3d origin_R0=R2ypr( pose0.rotationMatrix());
  Vector3d origin_R00=R2ypr(new_pose);
 double y_diff = origin_R0.x() - origin_R00.x();
   Matrix3d rot_diff = ypr2R(Vector3d(y_diff, 0, 0));
   if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            rot_diff =pose0.rotationMatrix() *new_pose.inverse();
        }
    for(auto iter:keyframes)
    {
        Frame::Ptr keyframe=iter .second;
        keyframe->SetPose(rot_diff*R*keyframe->pose.rotationMatrix(),rot_diff*R*(keyframe->pose.translation()-pose0.translation())+pose0.translation());
        keyframe->Vw=rot_diff*R*keyframe->Vw;
    }
}

bool  Initializer::InitializeIMU(Frames keyframes,double priorA,double priorG)
{
    double minTime=20.0;  // 初始化需要的最小时间间隔
    // 按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > KeyFrames_list;
    Frames::reverse_iterator   iter;
        for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++)
    {
        KeyFrames_list.push_front(iter->second);
    }
    std::vector< Frame::Ptr > Key_frames(KeyFrames_list.begin(),KeyFrames_list.end());

    const int N = Key_frames.size();  // 待处理的关键帧数目

    // 估计KF速度和重力方向
    if(!estimate_Vel_Rwg(Key_frames))
    {
        return false;
    }
    if(priorA==0){
        InertialOptimization(keyframes,Rwg,1e1,1e4);
    }
    else{
        InertialOptimization(keyframes,Rwg, priorG, priorA);
    }
        Vector3d dirG;
        dirG<< 0, 0, G;
        dirG=Rwg*dirG;
        dirG = dirG/dirG.norm();
        if(!(dirG[0]==0&&dirG[1]==0&&dirG[2]==1)){
            Vector3d  gI(0.0, 0.0, 1.0);//沿-z的归一化的重力数值
            // 计算旋转轴
            Vector3d v = gI.cross(dirG);
            const double nv = v.norm();
            // 计算旋转角
            const double cosg = gI.dot(dirG);
            const double ang = acos(cosg);
            // 计算mRwg，与-Z旋转偏差
            Vector3d vzg = v*ang/nv;
            Rwg = ExpSO3(vzg);
        }
        Vector3d g2;
        g2<< 0, 0, G;
        g2=Rwg*g2;
     LOG(INFO)<<"OPTG "<<(g2).transpose();

   if(bimu==false||reinit==true){
        Map::Instance().ApplyScaledRotation(Rwg.inverse());
   }
   else{
        ApplyScaledRotation(Rwg.inverse(),keyframes);
   }
    for(int i=0;i<N;i++)
    {
        Frame::Ptr pKF2 = Key_frames[i];
        pKF2->bImu = true;
    }

    if(priorA==0){
        FullInertialBA2(keyframes);
    }
    else{
        FullInertialBA(keyframes, priorG, priorA);
    }

    bimu=true;
    initialized=true;
    reinit=false;
    return true;
}

} // namespace lvio_fusioni
