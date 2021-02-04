#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{

void InertialOptimization(Frames &key_frames, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, double priorG, double priorA)   
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
    bg << para_gyroBias[0],para_gyroBias[1],para_gyroBias[2];
    ba <<para_accBias[0],para_accBias[1],para_accBias[2];
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
        }
}

void FullInertialBA(Frames &key_frames, Matrix3d Rwg, double priorG=1e2, double priorA=1e6)
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

    //先验BIAS约束
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
           //showIMUErrorINIT(para_kf_last, para_v_last,  para_gyroBias,para_accBias, para_kf, para_v,current_frame->preintegration,current_frame->time-1.40364e+09);

        }
        last_frame = current_frame;
    }
    //epa  para_accBias

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    //options.max_solver_time_in_seconds =0.1;
    options.max_num_iterations=4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.BriefReport();
    //数据恢复
      //  recoverData(key_frames,old_pose);
    Bias lastBias;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if(current_frame->bImu&&current_frame->last_keyframe)
        {
            Bias bias(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
            current_frame->SetNewBias(bias);
            lastBias=bias;
            //  LOG(INFO)<<"FullInertialBA  "<<current_frame->time-1.40364e+09+8.60223e+07<<"   Vwb1  "<<current_frame->Vw.transpose();
            //   LOG(INFO)<<"BIAS   a "<<bias.linearized_ba.transpose()<<" g "<<bias.linearized_bg.transpose();
        }
    }
   
    return ;
}




Bias FullInertialBA2(Frames &key_frames, Matrix3d Rwg)
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
    //epa  para_accBias

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    //options.max_solver_time_in_seconds =0.01;
    options.max_num_iterations=4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.BriefReport();
    //数据恢复
    
    recoverData(key_frames,old_pose);
    Bias lastBias;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if(current_frame->bImu&&current_frame->last_keyframe)
        {
            lastBias=current_frame->GetImuBias();
             LOG(INFO)<<"FullInertialBA  "<<current_frame->time-1.40364e+09+8.60223e+07<<"   Vwb1  "<<current_frame->Vw.transpose();
              LOG(INFO)<<"BIAS   a "<<current_frame->GetImuBias().linearized_ba.transpose()<<" g "<<current_frame->GetImuBias().linearized_bg.transpose();
        }
    }
   
    return lastBias;
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
        // bg = Map::Instance().current_frame->GetGyroBias();
        // ba =Map::Instance().current_frame->GetAccBias();
    }
    return true;
}

Vector3d Initializer::ComputeGyroBias(const Frames &frames)
{
    const int N = frames.size();
    std::vector<double> vbx;
    vbx.reserve(N);
    std::vector<double> vby;
    vby.reserve(N);
    std::vector<double> vbz;
    vbz.reserve(N);

    Matrix3d H = Matrix3d::Zero();
    Vector3d grad = Vector3d::Zero();
    bool first=true;
    Frame::Ptr pF1;
    for(auto iter:frames)
    {
        if(first){
            pF1=iter.second;
           first=false; 
            continue;
        }
        Frame::Ptr pF2 = iter.second;
        Matrix3d VisionR = pF1->GetImuRotation().transpose()*pF2->GetImuRotation();
        Matrix3d JRg = pF2->preintegration-> jacobian.block<3, 3>(3, 12);
        Matrix3d E = pF2->preintegrationFrame->GetUpdatedDeltaRotation().inverse().toRotationMatrix()*VisionR;
        Vector3d e =  LogSO3(E);
       // assert(fabs(pF2->time-pF1->time-pF2->preintegration->dT)<0.01);

       Matrix3d J = -InverseRightJacobianSO3(e)*E.transpose()*JRg;
        grad += J.transpose()*e;
        H += J.transpose()*J;
        pF1=iter.second;
    }

    Vector3d bg = -H.ldlt().solve( grad);
    LOG(INFO)<<bg.transpose();
    for(auto iter:frames)
        {
            Frame::Ptr pF = iter.second;
            pF->ImuBias.linearized_ba=Vector3d::Zero();
            pF->ImuBias.linearized_bg=bg;
            pF->SetNewBias(pF->ImuBias);
        }
    return bg;
}

Vector3d Initializer::ComputeVelocitiesAccBias(const Frames &frames)
{
   const int N = frames.size();
    const int nVar = 3*N +3; // 3 velocities/frame + acc bias
    const int nEqs = 6*(N-1);

    MatrixXd J(nEqs,nVar);
    J.setZero();
    VectorXd e(nEqs);
    e.setZero();
    Vector3d g;
    g<<0,0,-9.81007;

    int i=0;
    bool first=true;
    Frame::Ptr Frame1;
    Frame::Ptr Frame2;
    for(auto iter:frames)
    {
        if(first){
            Frame1=iter.second;
            first=false;
            continue;
        }

        Frame2=iter.second;
        Vector3d twb1 = Frame1->GetImuPosition();
        Vector3d twb2 = Frame2->GetImuPosition();
        Matrix3d Rwb1 = Frame1->GetImuRotation();
        Vector3d dP12 = Frame2->preintegration->GetUpdatedDeltaPosition();
        Vector3d dV12 = Frame2->preintegration->GetUpdatedDeltaVelocity();
        Matrix3d JP12 = Frame2->preintegration-> jacobian.block<3, 3>(0, 9);
        Matrix3d JV12 = Frame2->preintegration->jacobian.block<3, 3>(6, 9);
        float t12 = Frame2->preintegration->sum_dt;
        // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
        J.block<3,3>(6*i,3*i)+= Matrix3d::Identity()*t12;
        J.block<3,3>(6*i,3*N) += Rwb1*JP12;
        e.block<3,1>(6*i,0) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
        // Velocity v2=v1+g*t+R1*dV12
        J.block<3,3>(6*i+3,3*i)+= -Matrix3d::Identity();
        J.block<3,3>(6*i+3,3*i+3)+=  Matrix3d::Identity();
        J.block<3,3>(6*i+3,3*N) -= Rwb1*JV12;
        e.block<3,1>(6*i+3,0) = g*t12+Rwb1*dV12;
        Frame1=Frame2;
        i++;
    }

    MatrixXd H = J.transpose()*J;
    MatrixXd B = J.transpose()*e;
     VectorXd x=H.ldlt().solve(B);
     Vector3d ba;
    ba(0) = x(3*N);
    ba(1) = x(3*N+1);
    ba(2) = x(3*N+2);
 
    i=0;
    for(auto iter:frames)
    {
        Frame::Ptr pF = iter.second;
        pF->preintegration->Repropagate(Vector3d::Zero(),bg);
        pF->Vw=x.block<3,1>(3*i,0);
        if(i>0)
        {
            pF->ImuBias.linearized_ba=ba;
            pF->SetNewBias(pF->ImuBias);
        }
        i++;
    }
    return ba;
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
        InertialOptimization(keyframes,Rwg, Scale, bg, ba, 1e1,1e4);
    }
    else{
        InertialOptimization(keyframes,Rwg, Scale, bg, ba, priorG, priorA);
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
      FullInertialBA2(keyframes,Rwg);
    }
    else{
      FullInertialBA(keyframes,Rwg, priorG, priorA);
    }

   // frontend_.lock()->last_key_frame->bImu = true;
    bimu=true;
    initialized=true;
    reinit=false;
    return true;
}

} // namespace lvio_fusioni
