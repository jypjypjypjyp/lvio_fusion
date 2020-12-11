#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{
void InertialOptimization(unsigned long last_initialized_id, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, Eigen::MatrixXd  &covInertial, bool bFixedVel=false, bool bGauss=false, double priorG = 1, double priorA = 1e9)
{

     Frames vpKFs = Map::Instance().GetAllKeyFrames();
     ceres::Problem problem;
     auto para_gyroBias=vpKFs.begin()->second->mImuBias.linearized_bg.data();
     problem.AddParameterBlock(para_gyroBias, 3);
     ceres::CostFunction *cost_function = PriorGyroError::Create(Vector3d::Zero(),priorG);
     problem.AddResidualBlock(cost_function,NULL,para_gyroBias);

     auto para_accBias=vpKFs.begin()->second->mImuBias.linearized_ba.data();
     problem.AddParameterBlock(para_accBias, 3);
     cost_function = PriorAccError::Create(Vector3d::Zero(),priorA);
     problem.AddResidualBlock(cost_function,NULL,para_accBias);
    

     Vector3d rwg=Rwg.eulerAngles(0,1,2);
      auto para_rwg=rwg.data();
     problem.AddParameterBlock(para_rwg, 3);
     Frame::Ptr last_frame_;
     Frame::Ptr current_frame_;
     for(Frames::iterator iter = vpKFs.begin(); iter != vpKFs.end(); iter++)
     {
          current_frame_=iter->second;
          if(current_frame_->id>last_initialized_id)break;
          double timestamp=iter->first;
          if (!current_frame_->bImu||!current_frame_->mpLastKeyFrame||!current_frame_->preintegration->isPreintegrated)
          {
                last_frame_=current_frame_;
               continue;
          }
          auto para_v = current_frame_->mVw.data();
          problem.AddParameterBlock(para_v, 3);   

          if (last_frame_ && last_frame_->bImu&&last_frame_->mpLastKeyFrame)
          {
               auto para_v_last = last_frame_->mVw.data();
               cost_function = InertialGSError::Create(current_frame_->preintegration,current_frame_->pose,last_frame_->pose);
               problem.AddResidualBlock(cost_function, NULL,para_v_last,  para_v,para_gyroBias,para_accBias,para_rwg);
          }
          last_frame_ = current_frame_;
     }

     ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.6;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
     std::this_thread::sleep_for(std::chrono::seconds(3));

     Eigen::AngleAxisd rollAngle(AngleAxisd(rwg(0),Vector3d::UnitX()));
     Eigen::AngleAxisd pitchAngle(AngleAxisd(rwg(1),Vector3d::UnitY()));
     Eigen::AngleAxisd yawAngle(AngleAxisd(rwg(2),Vector3d::UnitZ()));
     Rwg= yawAngle*pitchAngle*rollAngle;
//for kfs  setNewBias 
     Bias bias_(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
     bg << para_gyroBias[0],para_gyroBias[1],para_gyroBias[2];
     ba <<para_accBias[0],para_accBias[1],para_accBias[2];
     for(Frames::iterator iter = vpKFs.begin(); iter != vpKFs.end(); iter++)
     {
           current_frame_=iter->second;
           if(current_frame_->id>last_initialized_id)break;
           Vector3d dbg=current_frame_->GetGyroBias()-bg;
          if(dbg.norm() >0.01)
          {
               current_frame_->SetNewBias(bias_);
               if (current_frame_->bImu)
                    current_frame_->preintegration->Reintegrate();
          }
          else
          {
               current_frame_->SetNewBias(bias_);
          }     
     }
}
    
void FullInertialBA(unsigned long last_initialized_id,Matrix3d Rwg, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0, bool *pbStopFlag=NULL, bool bInit=false, double priorG = 1, double priorA=1e9, Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL)
{
Frames KFs=Map::Instance().GetAllKeyFrames();
ceres::Problem problem;
 ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));


Frame::Ptr pIncKF=KFs.begin()->second;
double* para_gyroBias;
double* para_accBias;
para_gyroBias=pIncKF->mImuBias.linearized_bg.data();
problem.AddParameterBlock(para_accBias, 3);   
para_accBias=pIncKF->mImuBias.linearized_ba.data();
problem.AddParameterBlock(para_accBias, 3);   

Frame::Ptr last_frame;
Frame::Ptr current_frame;
int i=KFs.size();
for (auto kf_pair : KFs)
{
    i--;
    current_frame = kf_pair.second;
    if (!current_frame->bImu||!current_frame->mpLastKeyFrame||!current_frame->preintegration->isPreintegrated)
    {
        last_frame=current_frame;
        continue;
    }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->mVw.data();
            problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->bImu&&last_frame->mpLastKeyFrame)
        {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->mVw.data();
                ceres::CostFunction *cost_function = ImuError2::Create(current_frame->preintegration,Rwg);
                problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_accBias, para_gyroBias, para_kf, para_v);
                // LOG(INFO)<<"ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id;
                // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose();
                // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
                // LOG(INFO)<<"     current velocity: "<<current_frame->mVw.transpose();
                // LOG(INFO)<<"     last  velocity: "<<last_frame->mVw.transpose();
        }
        last_frame = current_frame;
    }
     //epa  para_accBias
     ceres::CostFunction *cost_function = PriorAccError::Create(Vector3d::Zero(),priorA);
     problem.AddResidualBlock(cost_function, NULL, para_accBias);//3
     //epg para_gyroBias
     cost_function = PriorGyroError::Create(Vector3d::Zero(),priorG);
     problem.AddResidualBlock(cost_function, NULL, para_gyroBias);//3

//solve
     ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.8;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//数据恢复
for(Frames::iterator iter = KFs.begin(); iter != KFs.end(); iter++)
{
     current_frame=iter->second;
     if(current_frame->id>last_initialized_id)break;
     if(current_frame->bImu&&current_frame->mpLastKeyFrame)
     {
          Bias b(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
          current_frame->SetNewBias(b);
     }
}
}

void  Initializer::InitializeIMU(bool bFIBA)
{
     double priorA=1e3;
     double priorG=1;
    double minTime=1.0;  // 初始化需要的最小时间间隔
    int nMinKF=10;     // 初始化需要的最少关键帧数
    if(Map::Instance().GetAllKeyFramesSize()<nMinKF)
        return;
   // Step 1:按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > lpKF;
    Frames keyframes=Map::Instance().GetAllKeyFrames();
    Frames::reverse_iterator   iter;
     for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++){
          lpKF.push_front(iter->second);
     }
    std::vector< Frame::Ptr > vpKF(lpKF.begin(),lpKF.end());
    if(vpKF.size()<nMinKF)
        return;
     unsigned long last_initialized_id=vpKF.back()->id;
     LOG(INFO)<<"last_initialized_id"<<last_initialized_id;
       // imu计算初始时间
    mFirstTs=vpKF.front()->time;
    if(Map::Instance().current_frame->time-mFirstTs<minTime)
        return;

    //bInitializing = true;   // 暂时未使用

    const int N = vpKF.size();  // 待处理的关键帧数目
    Bias b(0,0,0,0,0,0);

    // Step 2:估计KF速度和重力方向
    if (!initialized)
    {
        Vector3d dirG = Vector3d::Zero();  // 重力方向
        for(std::vector<Frame::Ptr>::iterator itKF = vpKF.begin()+1; itKF!=vpKF.end(); itKF++) 
        {
             if((*itKF)->id>last_initialized_id)break;
            if (!(*itKF)->preintegration)
                continue;
            if(!(*itKF)->mpLastKeyFrame)
               continue;
            // 预积分中delta_V 用来表示:Rwb_i.transpose()*(V2 - V1 - g*dt),故此处获得 -(V_i - V_0 - (i-0)*(mRwg*gI)*dt)
            // 应该使用将 (*itKF)->mpLastKeyFrame速度偏差在此处忽略或当做噪声，因为后面会优化mRwg
            dirG -= (*itKF)->mpLastKeyFrame->GetImuRotation()*(*itKF)->preintegration->GetUpdatedDeltaVelocity();
            Vector3d _vel = ((*itKF)->GetImuPosition() - (*(itKF-1))->GetImuPosition())/(*itKF)->preintegration->dT;
          if((*itKF)->preintegration->dT!=0){
               (*itKF)->SetVelocity(_vel);
               (*itKF)->mpLastKeyFrame->SetVelocity(_vel);
            }
        }

        // Step 2.1:计算重力方向(与z轴偏差)，用轴角方式表示偏差
        dirG = dirG/dirG.norm();
         Vector3d  gI(0.0, 0.0, -1.0);//沿-z的归一化的重力数值
        // dirG和gI的模长都是1,故cross为sin，dot为cos
        
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v*ang/nv;
        mRwg = ExpSO3(vzg);
        mTinit = Map::Instance().current_frame->time-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = Map::Instance().current_frame->GetGyroBias();
        mba =Map::Instance().current_frame->GetAccBias();
    }
 
    mScale=1.0;

    
    mInitTime = frontend_.lock()->last_frame->time-vpKF.front()->time;
    
    // Step 3:进行惯性优化

    // 使用camera初始地图frame的pose与预积分的差值优化
    Eigen::MatrixXd infoInertial=Eigen::MatrixXd::Zero(9,9);
    LOG(INFO)<<"InertialOptimization++++++++++++++";
    InertialOptimization(last_initialized_id,mRwg, mScale, mbg, mba,infoInertial , false, false, priorG, priorA);
     LOG(INFO)<<"InertialOptimization-------------------";

    // 如果求解的scale过小，跳过，下次在优化
    if (mScale<1e-1)
    {
        //bInitializing=false;
        return;
    }

    // 上面的程序没有改变地图，下面会对地图进行修改

    // Step 4:更新地图

    //map_->ApplyScaledRotation(mRwg.transpose(),mScale);
    frontend_.lock()->UpdateFrameIMU(mScale,vpKF[0]->GetImuBias(),frontend_.lock()->current_key_frame);

    // Check if initialization OK
    // Step 4:更新关键帧中imu状态
        for(int i=0;i<N;i++)
        {
            Frame::Ptr pKF2 = vpKF[i];
            pKF2->bImu = true;
        }

    // Step 5: 进行完全惯性优化(包括MapPoints)
LOG(INFO)<<"FullInertialBA+++++++++++++++++++++";
    FullInertialBA(last_initialized_id,mRwg,100, false, 0, NULL, true, priorG, priorA);
LOG(INFO)<<"FullInertialBA------------------------------";

    // Step 6: 设置当前map imu 已经初始化 
    // If initialization is OK
   frontend_.lock()->UpdateFrameIMU(1.0,vpKF[last_initialized_id-1]->GetImuBias(),frontend_.lock()->current_key_frame);

    frontend_.lock()->current_key_frame->bImu = true;
   // }
    //更新记录初始化状态的变量
    mbNewInit=true;
    mnKFs=vpKF.size();
    mIdxInit++;

    //bInitializing = false;
    bimu=true;
    return;
}

} // namespace lvio_fusion