#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{
Bias InertialOptimization(Frames key_frames, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, double priorG, double priorA)
{
    ceres::Problem problem;

    //先验BIAS约束
    auto para_gyroBias=key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);
    ceres::CostFunction *cost_function = PriorGyroError2::Create(Vector3d::Zero(),priorG);
    problem.AddResidualBlock(cost_function,NULL,para_gyroBias);

    auto para_accBias=key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);
    cost_function = PriorAccError2::Create(Vector3d::Zero(),priorA);
    problem.AddResidualBlock(cost_function,NULL,para_accBias);
    
    //优化重力、BIAS和速度的边
    Quaterniond rwg(Rwg);
    SO3d RwgSO3(rwg);
    auto para_rwg=RwgSO3.data();
        ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(para_rwg, SO3d::num_parameters,local_parameterization);
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    double *para_scale=&scale;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if (!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_v, 3);   

        if (last_frame)
        {
            auto para_v_last = last_frame->Vw.data();
            cost_function = InertialGSError2::Create(current_frame->preintegration,current_frame->pose,last_frame->pose);
            problem.AddResidualBlock(cost_function, NULL,para_v_last,  para_v,para_gyroBias,para_accBias,para_rwg);
                // LOG(INFO)<<"InertialOptimization ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id<<"  t"<<current_frame->preintegration->dT;
                // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose(); 
                // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
                // LOG(INFO)<<"     current velocity: "<<current_frame->Vw.transpose();
                // LOG(INFO)<<"     last  velocity: "<<last_frame->Vw.transpose();
        }
        last_frame = current_frame;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.max_solver_time_in_seconds = 4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::this_thread::sleep_for(std::chrono::seconds(3));

    //数据恢复
    // Eigen::AngleAxisd rollAngle(AngleAxisd(rwg(0),Vector3d::UnitX()));
    // Eigen::AngleAxisd pitchAngle(AngleAxisd(rwg(1),Vector3d::UnitY()));
    // Eigen::AngleAxisd yawAngle(AngleAxisd(rwg(2),Vector3d::UnitZ()));
    // Rwg= yawAngle*pitchAngle*rollAngle;
    Quaterniond rwg2(RwgSO3.data()[0],RwgSO3.data()[1],RwgSO3.data()[2],RwgSO3.data()[3]);
        Rwg=rwg2.toRotationMatrix();
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
                        Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
    //   LOG(INFO)<<"OPTG "<<(g).transpose()<<"Rwg"<<RwgSO3.data()[0]<<" "<<RwgSO3.data()[1]<<" "<<RwgSO3.data()[2]<<" "<<RwgSO3.data()[3];
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
            current_frame->preintegration->Reintegrate();
        }
       else
        {
            current_frame->SetNewBias(bias_);
        }     
            Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
        LOG(INFO)<<"InertialOptimization  "<<current_frame->time-1.40364e+09/*+8.60223e+07*/<<"   Vwb1  "<<current_frame->Vw.transpose()*tcb;//<<"\nR: \n"<<tcb.inverse()*current_frame->pose.rotationMatrix();
    }
    
    return bias_;
}

Bias FullInertialBA(Frames key_frames, Matrix3d Rwg, double priorG, double priorA)
{
    ceres::Problem problem;
    
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = key_frames.begin()->first;
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
            ceres::CostFunction *cost_function;
            if (first_frame->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else if (first_frame != frame)
            {
                double *para_fist_kf = first_frame->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
            }
        }
    }

    Frame::Ptr pIncKF=key_frames.begin()->second;
    double* para_gyroBias;
    double* para_accBias;
    para_gyroBias=pIncKF->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_accBias, 3);   
    para_accBias=pIncKF->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);   

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    int i=key_frames.size();
    for (auto kf_pair : key_frames)
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
        //problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->bImu)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            ceres::CostFunction *cost_function =InertialError2::Create(current_frame->preintegration,Rwg);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last,  para_gyroBias,para_accBias, para_kf, para_v);
                // LOG(INFO)<<"FullInertialBA ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id;
                // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose(); 
                // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
                // LOG(INFO)<<"     current velocity: "<<current_frame->Vw.transpose();
                // LOG(INFO)<<"     last  velocity: "<<last_frame->Vw.transpose();
        }
        last_frame = current_frame;
    }
    //epa  para_accBias
    ceres::CostFunction *cost_function = PriorAccError2::Create(Vector3d::Zero(),priorA);
    problem.AddResidualBlock(cost_function, NULL, para_accBias);
    //epg para_gyroBias
    cost_function = PriorGyroError2::Create(Vector3d::Zero(),priorG);
    problem.AddResidualBlock(cost_function, NULL, para_gyroBias);
    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.max_solver_time_in_seconds = 4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
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
               Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
            //  LOG(INFO)<<"FullInertialBA  "<<current_frame->time-1.40364e+09/*+8.60223e+07*/<<"   Vwb1  "<<current_frame->Vw.transpose()*tcb<<"\nR: \n"<<tcb.inverse()*current_frame->pose.rotationMatrix();
        }
    }
    return lastBias;
}

bool  Initializer::estimate_Vel_Rwg(std::vector< Frame::Ptr > Key_frames)
{

               Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
    if (!initialized)
    {
        Vector3d dirG = Vector3d::Zero();
       // bool isfirst=true;
        Vector3d velocity;
        for(std::vector<Frame::Ptr>::iterator iter_keyframe = Key_frames.begin()+1; iter_keyframe!=Key_frames.end(); iter_keyframe++) 
        {
            if(!(*iter_keyframe)->last_keyframe)
            {
                continue;
            }
            if(!(*iter_keyframe)->preintegration->isPreintegrated)
            {
                return false;
            }   

            dirG -= (*iter_keyframe)->last_keyframe->GetImuRotation()*(*iter_keyframe)->preintegration->GetUpdatedDeltaVelocity();
            velocity= ((*iter_keyframe)->GetImuPosition() - (*(iter_keyframe))->last_keyframe->GetImuPosition())/((*iter_keyframe)->preintegration->dT);
            (*iter_keyframe)->SetVelocity(velocity);
            (*iter_keyframe)->last_keyframe->SetVelocity(velocity);
                
            LOG(INFO)<<"InitializeIMU  "<<(*iter_keyframe)->time-1.40364e+09/*+8.60223e+07*/<<"   Vwb1  "<<(tcb.inverse()*velocity).transpose()<<"  dt " <<(*iter_keyframe)->preintegration->dT;
           // LOG(INFO)<<"current  "<<(*iter_keyframe)->id<<"   last  "<< (*(iter_keyframe))->last_keyframe->id;
            LOG(INFO)<<"current  "<<(tcb.inverse()*(*iter_keyframe)->GetImuPosition()).transpose()<<"   last  "<< (tcb.inverse()*(*(iter_keyframe))->last_keyframe->GetImuPosition()).transpose();
            //LOG(INFO)<<"  dP "<<(*iter_keyframe)->preintegration->dP.transpose()<<"  dV "<<(*iter_keyframe)->preintegration->dV.transpose();
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
        //Rwg=Matrix3d::Identity();
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
        // LOG(INFO)<<"dirG "<<(tcb.inverse()*dirG).transpose();
        LOG(INFO)<<"INITG "<<(tcb.inverse()*g).transpose();
    } 
    else
    {
        bg = Map::Instance().current_frame->GetGyroBias();
        ba =Map::Instance().current_frame->GetAccBias();
    }
    return true;
}


bool  Initializer::InitializeIMU(Frames keyframes)
{
    
    double priorA=1e5;
    double priorG=1e2;
    double minTime=20.0;  // 初始化需要的最小时间间隔
    // 按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > KeyFrames_list;
    Frames::reverse_iterator   iter;
    for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++)
    {
        KeyFrames_list.push_front(iter->second);
    }
    std::vector< Frame::Ptr > Key_frames(KeyFrames_list.begin(),KeyFrames_list.end());
    unsigned long last_initialized_id=Key_frames.back()->id;
    // imu计算初始时间
    FirstTs=Key_frames.front()->time;
    // LOG(INFO)<<FirstTs-Map::Instance().keyframes.begin()->second->time;
    // if(FirstTs-Map::Instance().keyframes.begin()->second->time<minTime)
    //     return false;

    const int N = Key_frames.size();  // 待处理的关键帧数目

    // 估计KF速度和重力方向
    if(!estimate_Vel_Rwg(Key_frames))
    {
        return false;
    }
 
    // KF速度和重力方向优化
    Scale=1.0;
                Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
                        Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
            LOG(INFO)<<"INITG "<<(tcb.inverse()*g).transpose()<<"\n"<<Rwg;
    Bias bias_=InertialOptimization(keyframes,Rwg, Scale, bg, ba, priorG, priorA);
        Vector3d dirG;
        //  dirG<< 0, 0, -G;
        //     dirG=Rwg*dirG;
        dirG<<-0.15915923848354485 ,  9.1792672332077689,   3.4306621858045498;
        dirG=tcb*dirG;
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
                        Vector3d g2;
         g2<< 0, 0, -G;
         g2=Rwg*g2;
            LOG(INFO)<<"OPTG "<<(tcb.inverse()*g2).transpose()<<"\n"<<Rwg;

    if (Scale<1e-1)
    {
        return false;
    }
    Map::Instance().ApplyScaledRotation(Rwg.inverse());
    // frontend_.lock()->UpdateFrameIMU(bias_);
    //更新关键帧中imu状态
    for(int i=0;i<N;i++)
    {
        Frame::Ptr pKF2 = Key_frames[i];
        pKF2->bImu = true;
    }

    // 进行完全惯性优化
  //  Bias lastBias=FullInertialBA(keyframes,Rwg, priorG, priorA);
    // frontend_.lock()->UpdateFrameIMU(lastBias);
    frontend_.lock()->current_key_frame->bImu = true;
    bimu=true;
    initialized=true;
    return true;
}

} // namespace lvio_fusioni
