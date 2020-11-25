#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{
void InertialOptimization(Map::Ptr pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, Eigen::MatrixXd  &covInertial, bool bFixedVel=false, bool bGauss=false, double priorG = 1e2, double priorA = 1e6)
{

     Frames vpKFs = pMap->GetAllKeyFrames();

     ceres::Problem problem;
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));


     Vector3d g=vpKFs.begin()->second->GetGyroBias();
     auto para_gyroBias=g.data();
     problem.AddParameterBlock(para_gyroBias, 3);
     ceres::CostFunction *cost_function = PriorGyroError::Create(Vector3d::Zero());
     problem.AddResidualBlock(cost_function,NULL,para_gyroBias);

     Vector3d a=vpKFs.begin()->second->GetAccBias();
     auto para_accBias=a.data();
     problem.AddParameterBlock(para_accBias, 3);
    cost_function = PriorAccError::Create(Vector3d::Zero());
     problem.AddResidualBlock(cost_function,NULL,para_accBias);
    

     Vector3d rwg=Rwg.eulerAngles(0,1,2);
      auto para_rwg=rwg.data();
     problem.AddParameterBlock(para_rwg, 3);
     problem.AddParameterBlock(&scale, 1);

     if (bFixedVel)//false
     {
          problem.SetParameterBlockConstant(para_gyroBias);
          problem.SetParameterBlockConstant(para_accBias);
     }
     problem.SetParameterBlockConstant(&scale);

     Frame::Ptr last_frame;
     Frame::Ptr current_frame;
     for(Frames::iterator iter = vpKFs.begin(); iter != vpKFs.end(); iter++)
     {
          current_frame=iter->second;
          double timestamp=iter->first;
          if (!current_frame->preintegration)
               continue;
          auto para_kf = current_frame->pose.data();
          auto para_v = current_frame->mVw.data();
     
          problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
          problem.AddParameterBlock(para_v, 3);   
          problem.SetParameterBlockConstant(para_kf);
          if (bFixedVel)
               problem.SetParameterBlockConstant(para_v);

          if (last_frame && current_frame->preintegration)
          {
               auto para_kf_last = last_frame->pose.data();
               auto para_v_last = last_frame->mVw.data();
               cost_function = InertialGSError2::Create(current_frame->preintegration);
               problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_kf, para_v,para_gyroBias,para_accBias,para_rwg,&scale);
          }
          last_frame = current_frame;
     }

     ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-9;
    options.max_solver_time_in_seconds = 7 * 0.6;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


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
           current_frame=iter->second;
           Vector3d dbg=current_frame->GetGyroBias()-bg;
          if(dbg.norm() >0.01)
          {
               current_frame->SetNewBias(bias_);
               if (current_frame->preintegration)
                    current_frame->preintegration->Reintegrate();
          }
          else
          {
               current_frame->SetNewBias(bias_);
          }
          
     }

}
    
void FullInertialBA(Map::Ptr pMap, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0, bool *pbStopFlag=NULL, bool bInit=false, double priorG = 1e2, double priorA=1e6, Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL)
{
auto KFs=pMap->GetAllKeyFrames();
//auto MPs=pMap->GetAllLandmarks();
ceres::Problem problem;
 ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));


std::vector<double *> para_kfs;
std::vector<double *> para_vs;
std::vector<double *> para_gbs;
std::vector<double *> para_abs;
para_kfs.reserve(KFs.size());
para_vs.reserve(KFs.size());
para_gbs.reserve(KFs.size());
para_abs.reserve(KFs.size());

int nNonFixed = 0;
Frame::Ptr pIncKF;
int i=0;
for(Frames::iterator iter = KFs.begin(); iter != KFs.end(); iter++,i++)
{
     Frame::Ptr KFi=iter->second;
     auto para_kf=KFi->pose.data();
     problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);   
     para_kfs[i]=para_kf;
     pIncKF=KFi;
     if(KFi->bImu)
     {
          auto para_v = KFi->mVw.data();
          problem.AddParameterBlock(para_v, 3);   
          para_vs[i]=para_v;
          if(!bInit)
          {
                Vector3d g=KFi->GetGyroBias();
               auto para_gyroBias=g.data();
               para_gbs[i]=para_gyroBias;
               problem.AddParameterBlock(para_gyroBias, 3);   
               Vector3d a=KFi->GetAccBias();
               auto para_accBias=a.data();
               para_abs[i]=para_accBias;
               problem.AddParameterBlock(para_accBias, 3);   
          }
     }
}
double* para_gyroBias;
double* para_accBias;
if(bInit)
{
    Vector3d g=pIncKF->GetGyroBias();
     para_gyroBias=g.data();
      problem.AddParameterBlock(para_accBias, 3);   
     Vector3d a=pIncKF->GetAccBias();
     para_accBias=a.data();
      problem.AddParameterBlock(para_accBias, 3);   
}

int ii=0;
Frame::Ptr last_frame;
Frame::Ptr current_frame;
for(Frames::iterator iter = KFs.begin(); iter != KFs.end(); iter++,ii++)
{
     current_frame=iter->second;
     if(ii==0)
     {
          last_frame=current_frame;
          continue;
     }
     if(current_frame->bImu && last_frame->bImu&&current_frame->preintegration)
     {
          current_frame->SetNewBias(last_frame->GetImuBias());
          auto P1=para_kfs[ii-1];
          auto V1=para_vs[ii-1];
          double* g1,*g2,*a1,*a2;
          if(!bInit){
               g1=para_gbs[ii-1];
               a1=para_abs[ii-1];
               g2=para_gbs[ii];
               a2=para_abs[ii];
          }else
          {
               g1=para_gyroBias;
               a1=para_accBias;
          }
          auto P2=para_kfs[ii];
          auto V2=para_vs[ii];

          //ei p1,v1,g1,a1,p2,v2
           ceres::CostFunction *cost_function = InertialError2::Create(current_frame->preintegration);
           problem.AddResidualBlock(cost_function, NULL, P1,V1,g1,a1,P2,V2);//7,3,3,3,7,3


          if(!bInit){
               //egr g1 g2
               cost_function = GyroRWError::Create();
               problem.AddResidualBlock(cost_function, NULL, g1,g2);//3,3
               //ear a1 a2  
               cost_function = AccRWError::Create();
               problem.AddResidualBlock(cost_function, NULL, a1,a2);//3,3
          }
     }
     last_frame = current_frame;
}

//先验
if(bInit){
     //epa  para_accBias
     ceres::CostFunction *cost_function = PriorAccError::Create(Vector3d::Zero());
     problem.AddResidualBlock(cost_function, NULL, para_accBias);//3
     //epg para_gyroBias
     cost_function = PriorGyroError::Create(Vector3d::Zero());
     problem.AddResidualBlock(cost_function, NULL, para_gyroBias);//3
}

//solve
     ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-9;
    options.max_solver_time_in_seconds = 7 * 0.6;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//数据恢复
int iii=0;
 current_frame;
for(Frames::iterator iter = KFs.begin(); iter != KFs.end(); iter++,iii++)
{
     current_frame=iter->second;

     if(current_frame->bImu)
     {
          double* ab;
          double* gb;
          if(!bInit)
          {
               gb=para_gbs[iii];
               ab=para_abs[iii];
          }
          else
          {
               gb=para_gyroBias;
               ab=para_gyroBias;
          }
          Bias b(ab[0],ab[1],ab[2],gb[0],gb[1],gb[2]);
          current_frame->SetNewBias(b);
     }
}
}

void  Initializer::InitializeIMU(double priorG, double priorA, bool bFIBA)
{
    double minTime=1.0;  // 初始化需要的最小时间间隔
    int nMinKF=10;     // 初始化需要的最少关键帧数
    if(map_->GetAllKeyFramesSize()<nMinKF)
        return;
   // Step 1:按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > lpKF;
    Frames keyframes=map_->GetAllKeyFrames();
    Frames::reverse_iterator   iter;
     for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++){
          lpKF.push_front(iter->second);
     }
    std::vector< Frame::Ptr > vpKF(lpKF.begin(),lpKF.end());
    if(vpKF.size()<nMinKF)
        return;
    
       // imu计算初始时间
    mFirstTs=vpKF.front()->time;
    if(map_->current_frame->time-mFirstTs<minTime)
        return;

    bInitializing = true;   // 暂时未使用

    const int N = vpKF.size();  // 待处理的关键帧数目
    Bias b(0,0,0,0,0,0);

    // Step 2:估计KF速度和重力方向
    if (!initialized)
    {
        Vector3d dirG = Vector3d::Zero();  // 重力方向
        for(std::vector<Frame::Ptr>::iterator itKF = vpKF.begin()+1; itKF!=vpKF.end(); itKF++) 
        {
            if (!(*itKF)->preintegration)
                continue;
            if(!(*itKF)->mpLastKeyFrame)
               continue;
            // 预积分中delta_V 用来表示:Rwb_i.transpose()*(V2 - V1 - g*dt),故此处获得 -(V_i - V_0 - (i-0)*(mRwg*gI)*dt)
            // 应该使用将 (*itKF)->mpLastKeyFrame速度偏差在此处忽略或当做噪声，因为后面会优化mRwg
            dirG -= (*itKF)->mpLastKeyFrame->GetImuRotation()*(*itKF)->preintegration->GetUpdatedDeltaVelocity();

            Vector3d _vel = ((*itKF)->GetImuPosition() - (*(itKF-1))->GetImuPosition())/(*itKF)->preintegration->dT;
          if(!(*itKF)->preintegration->dT==0){
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
        mTinit = map_->current_frame->time-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = map_->current_frame->GetGyroBias();
        mba = map_->current_frame->GetAccBias();
    }
 
    mScale=1.0;

    
    mInitTime = frontend_.lock()->last_frame->time-vpKF.front()->time;
    
    // Step 3:进行惯性优化

    // 使用camera初始地图frame的pose与预积分的差值优化
    Eigen::MatrixXd infoInertial=Eigen::MatrixXd::Zero(9,9);
    InertialOptimization(map_, mRwg, mScale, mbg, mba,infoInertial , false, false, priorG, priorA);


    // 如果求解的scale过小，跳过，下次在优化
    if (mScale<1e-1)
    {
    //    cout << "scale too small" << endl;
        bInitializing=false;
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

    if (bFIBA)
    {
        if (priorA!=0.f)
            FullInertialBA(map_, 100, false, 0, NULL, true, priorG, priorA);
        else
            FullInertialBA(map_, 100, false, 0, NULL, false);
    }

    // Step 6: 设置当前map imu 已经初始化 
    // If initialization is OK
   frontend_.lock()->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),frontend_.lock()->current_key_frame);

       frontend_.lock()->current_key_frame->bImu = true;
   // }
    //更新记录初始化状态的变量
    mbNewInit=true;
    mnKFs=vpKF.size();
    mIdxInit++;

    bInitializing = false;
    bimu=true;
    return;

}


} // namespace lvio_fusion