#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{
void InertialOptimization(Map::Ptr pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel=false, bool bGauss=false, float priorG = 1e2, float priorA = 1e6)
{

     Frames vpKFs = pMap->GetAllKeyFrames();

     ceres::Problem problem;
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));


     Vector3d g;
     cv::cv2eigen(vpKFs.begin()->second->GetGyroBias(),g);
     auto para_gyroBias=g.data();
     problem.AddParameterBlock(para_gyroBias, 3);
     ceres::CostFunction *cost_function = PriorGyroError::Create(cv::Mat::zeros(3,1,CV_32F));
     problem.AddResidualBlock(cost_function,NULL,para_gyroBias);

     Vector3d a;
     cv::cv2eigen(vpKFs.begin()->second->GetAccBias(),a);
     auto para_accBias=a.data();
     problem.AddParameterBlock(para_accBias, 3);
    cost_function = PriorAccError::Create(cv::Mat::zeros(3,1,CV_32F));
     problem.AddResidualBlock(cost_function,NULL,para_accBias);
    

     Vector3d rwg=Rwg.eulerAngles(2,1,0);
      auto para_rwg=rwg.data();
     problem.AddParameterBlock(para_rwg, 3);
     problem.AddParameterBlock(&scale, 1);

     if (bFixedVel)
     {
          problem.SetParameterBlockConstant(para_gyroBias);
          problem.SetParameterBlockConstant(para_accBias);
     }
     if(!bMono)
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

          if (last_frame && last_frame->preintegration)
          {
               auto para_kf_last = last_frame->pose.data();
               auto para_v_last = last_frame->mVw.data();
               cost_function = InertialGSError::Create(current_frame->preintegration);
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


     Eigen::AngleAxisd rollAngle(AngleAxisd(rwg(2),Vector3d::UnitX()));
     Eigen::AngleAxisd pitchAngle(AngleAxisd(rwg(1),Vector3d::UnitY()));
     Eigen::AngleAxisd yawAngle(AngleAxisd(rwg(0),Vector3d::UnitZ()));
     Rwg= yawAngle*pitchAngle*rollAngle;
//for kfs  setNewBias 
     Bias b(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
     bg << para_gyroBias[0],para_gyroBias[1],para_gyroBias[2];
     ba <<para_accBias[0],para_accBias[1],para_accBias[2];
     cv::Mat cvbg = toCvMat(bg);
     for(Frames::iterator iter = vpKFs.begin(); iter != vpKFs.end(); iter++)
     {
           current_frame=iter->second;
          if(cv::norm(current_frame->GetGyroBias()-cvbg)>0.01)
          {
               current_frame->SetNewBias(b);
               if (current_frame->preintegration)
                    current_frame->preintegration->Reintegrate();//TODO
          }
          else
          {
               current_frame->SetNewBias(b);
          }
          
     }

}
    
void FullInertialBA(Map::Ptr pMap, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0, bool *pbStopFlag=NULL, bool bInit=false, float priorG = 1e2, float priorA=1e6, Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL)
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
                Vector3d g;
               cv::cv2eigen(KFi->GetGyroBias(),g);
               auto para_gyroBias=g.data();
               para_gbs[i]=para_gyroBias;
               problem.AddParameterBlock(para_gyroBias, 3);   
               Vector3d a;
               cv::cv2eigen(KFi->GetAccBias(),a);
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
     Vector3d g;
     cv::cv2eigen(pIncKF->GetGyroBias(),g);
     para_gyroBias=g.data();
      problem.AddParameterBlock(para_accBias, 3);   
     Vector3d a;
     cv::cv2eigen(pIncKF->GetAccBias(),a);
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
     if(current_frame->bImu && last_frame->bImu)
     {
          current_frame->preintegration->SetNewBias(last_frame->GetImuBias());
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
           ceres::CostFunction *cost_function = InertialError::Create(current_frame->preintegration);
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
     ceres::CostFunction *cost_function = PriorAccError::Create(cv::Mat::zeros(3,1,CV_32F));
     problem.AddResidualBlock(cost_function, NULL, para_accBias);//3
     //epg para_gyroBias
     cost_function = PriorGyroError::Create(cv::Mat::zeros(3,1,CV_32F));
     problem.AddResidualBlock(cost_function, NULL, para_gyroBias);//3
}

/* 不优化地图点
//mappoint
   // const float thHuberMono = sqrt(5.991);
    const float thHuberStereo =、 sqrt(7.815);
     
for(visual::Landmarks::iterator iter = MPs.begin(); iter != MPs.end(); iter++,ii++)
{
     visual::Landmark::Ptr mp=iter->second;
     auto para_point=mp->position.data();
     visual::Features observations = mp->observations;
     //bool bAllFixed = true;
     for(visual::Features::iterator iter=observations.begin();iter != observations.end(); iter++)
     {
          std::weak_ptr<Frame> KFi=iter->second->frame;





          //vp=para_kf
          auto para_kf=KFi.lock()->pose.data();
          //e para_point para_kf
          ceres::CostFunction *cost_function = StereoError::Create();
          problem.AddResidualBlock(cost_function, NULL,para_point, para_kf);//3,3

     }
     
}
*/

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
     auto para_kf=para_kfs[iii];
     Quaterniond Qi(para_kf[3], para_kf[0], para_kf[1], para_kf[2]);
     
     Vector3d Pi(para_kf[4], para_kf[5], para_kf[6]);
     cv::Mat Tcw = toCvSE3(Qi.toRotationMatrix(),Pi);

     current_frame->SetPose(Tcw);
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

void  Initializer::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    float minTime=1.0;  // 初始化需要的最小时间间隔
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
        cv::Mat cvRwg;
        cv::Mat dirG = cv::Mat::zeros(3,1,CV_32F);  // 重力方向
        for(std::vector<Frame::Ptr>::iterator itKF = vpKF.begin()+1; itKF!=vpKF.end(); itKF++) 
        {
            if (!(*itKF)->preintegration)
                continue;
            // 预积分中delta_V 用来表示:Rwb_i.transpose()*(V2 - V1 - g*dt),故此处获得 -(V_i - V_0 - (i-0)*(mRwg*gI)*dt)
            // 应该使用将速度偏差在此处忽略或当做噪声，因为后面会优化mRwg
            dirG -= (*(itKF-1))->GetImuRotation()*(*itKF)->preintegration->GetUpdatedDeltaVelocity();
            cv::Mat _vel = ((*itKF)->GetImuPosition() - (*(itKF-1))->GetImuPosition())/(*itKF)->preintegration->dT;
            (*itKF)->SetVelocity(_vel);
            (*(itKF-1))->SetVelocity(_vel);
        }

        // Step 2.1:计算重力方向(与z轴偏差)，用轴角方式表示偏差
        dirG = dirG/cv::norm(dirG);
        cv::Mat gI = (cv::Mat_<float>(3,1) << 0.0f, 0.0f, -1.0f); //沿-z的归一化的重力数值
        // dirG和gI的模长都是1,故cross为sin，dot为cos
        
        // 计算旋转轴
        cv::Mat v = gI.cross(dirG);
        const float nv = cv::norm(v);
        // 计算旋转角
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        cv::Mat vzg = v*ang/nv;
        cvRwg = ExpSO3(vzg);
        mRwg = toMatrix3d(cvRwg);
        mTinit = map_->current_frame->time-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = toVector3d(map_->current_frame->GetGyroBias());
        mba = toVector3d(map_->current_frame->GetAccBias());
    }
 
    mScale=1.0;

    
    mInitTime = frontend_.lock()->last_frame->time-vpKF.front()->time;
    
    // Step 3:进行惯性优化
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // 使用camera初始地图frame的pose与预积分的差值优化
    Eigen::MatrixXd infoInertial=Eigen::MatrixXd::Zero(9,9);
    InertialOptimization(map_, mRwg, mScale, mbg, mba, false,infoInertial , false, false, priorG, priorA);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    /*cout << "scale after inertial-only optimization: " << mScale << endl;
    cout << "bg after inertial-only optimization: " << mbg << endl;
    cout << "ba after inertial-only optimization: " << mba << endl;*/

    // 如果求解的scale过小，跳过，下次在优化
    if (mScale<1e-1)
    {
    //    cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    // 上面的程序没有改变地图，下面会对地图进行修改

 //   unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // Step 4:更新地图

    map_->ApplyScaledRotation(toCvMat(mRwg).t(),mScale,true);
    frontend_.lock()->UpdateFrameIMU(mScale,vpKF[0]->GetImuBias(),frontend_.lock()->current_key_frame);

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // Check if initialization OK
    // Step 4:更新关键帧中imu状态
        for(int i=0;i<N;i++)
        {
            Frame::Ptr pKF2 = vpKF[i];
            pKF2->bImu = true;
        }

    /*cout << "Before GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;*/

    // Step 5: 进行完全惯性优化(包括MapPoints)
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            FullInertialBA(map_, 100, false, 0, NULL, true, priorG, priorA);
        else
            FullInertialBA(map_, 100, false, 0, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    // Step 6: 设置当前map imu 已经初始化 
    // If initialization is OK
   frontend_.lock()->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),frontend_.lock()->current_key_frame);
   // if (!mpAtlas->isImuInitialized())
  //  {
      //  cout << "IMU in Map " << mpAtlas->GetCurrentMap()->GetId() << " is initialized" << endl;
 //       mpAtlas->SetImuInitialized();
     //   mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;  // 设置imu初始化时间
       frontend_.lock()->current_key_frame->bImu = true;
   // }
    //更新记录初始化状态的变量
    mbNewInit=true;
    mnKFs=vpKF.size();
    mIdxInit++;

    // Step 7: 清除KF待处理列表中剩余的KF
  /*  for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();
*/
    //mpTracker->mState=Tracking::OK;
    bInitializing = false;
    bimu=true;

    /*cout << "After GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;
    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
    double t_update = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
    double t_viba = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
    cout << t_inertial_only << ", " << t_update << ", " << t_viba << endl;*/

 //   mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;

}


/*bool Initializer::Initialize(Frames kfs)
{
    // be perpare for initialization
    std::vector<Initializer::Frame> frames;
    for (auto kf_pair : kfs)
    {
        if (!kf_pair.second->preintegration)
            return false;
        Initializer::Frame frame;
        frame.preintegration = kf_pair.second->preintegration;
        frame.R = kf_pair.second->pose.inverse().rotationMatrix();
        frame.T = kf_pair.second->pose.inverse().translation();
        frame.Ba = kf_pair.second->preintegration->linearized_ba;
        frame.Bg = kf_pair.second->preintegration->linearized_bg;
        frames.push_back(frame);
    }

    SolveGyroscopeBias(frames);
    for (auto frame : frames)
    {
        frame.preintegration->Repropagate(Vector3d::Zero(), frame.Bg);
    }
    LOG(INFO) << "IMU Initialization failed.";
    initialized = true;
    return true;
    // //check imu observibility
    // Frames::iterator frame_it;
    // Vector3d sum_g;
    // for (frame_it = frames_.begin(); next(frame_it) != frames_.end(); frame_it++)
    // {
    //     double dt = frame_it->second->preintegration->sum_dt;
    //     Vector3d tmp_g = frame_it->second->preintegration->delta_v / dt;
    //     sum_g += tmp_g;
    // }
    // Vector3d aver_g;
    // aver_g = sum_g * 1.0 / ((int)frames_.size() - 1);
    // double var = 0;
    // for (frame_it = frames_.begin(); next(frame_it) != frames_.end(); frame_it++)
    // {
    //     double dt = frame_it->second->preintegration->sum_dt;
    //     Vector3d tmp_g = frame_it->second->preintegration->delta_v / dt;
    //     var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    // }
    // var = sqrt(var / ((int)frames_.size() - 1));
    // if (var < 0.25)
    // {
    //     LOG(INFO) << "IMU excitation not enouth!";
    //     return false;
    // }

    // // visual initial align
    // if (VisualInitialAlign())
    //     return true;
    // else
    // {
    //     LOG(INFO) << "misalign visual structure with IMU";
    //     return false;
    // }
}

// bool Initializer::VisualInitialAlign()
// {
//     VectorXd x;
//     //solve scale
//     bool result = VisualIMUAlignment(x);
//     if (!result)
//     {
//         //ROS_DEBUG("solve g_ failed!");
//         return false;
//     }

//     double s = (x.tail<1>())(0);
//     Frames::iterator frame_i;
//     for (frame_i = frames_.begin(); frame_i != frames_.end(); frame_i++)
//     {
//         frame_i->second->preintegration->Repropagate(Vector3d::Zero(), frame_i->second->preintegration->Bg);
//     }

//     Matrix3d R0 = g2R(g_);
//     double yaw = R2ypr(R0 * frames_.begin()->second->pose.rotationMatrix()).x();
//     R0 = ypr2R(Vector3d{-yaw, 0, 0}) * R0;
//     g_ = R0 * g_;

//     Matrix3d rot_diff = R0;

//     return true;
// }

void Initializer::SolveGyroscopeBias(std::vector<Initializer::Frame> &frames)
{
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();
    Vector3d delta_bg;

    for (int i = 0, j = 1; j < frames.size() - 1; i++, j++)
    {
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Quaterniond q_ij(frames[i].R.transpose() * frames[j].R);
        tmp_A = frames[i].preintegration->jacobian.template block<3, 3>(imu::O_R, imu::O_BG);
        tmp_b = 2 * (frames[i].preintegration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);

    for (int i = 0; i < frames.size() - 1; i++)
    {
        frames[i].Bg += delta_bg;
        frames[i].preintegration->Repropagate(Vector3d::Zero(), frames[0].Bg);
    }
}
*/
// inline MatrixXd TangentBasis(Vector3d &g0)
// {
//     Vector3d b, c;
//     Vector3d a = g0.normalized();
//     Vector3d tmp(0, 0, 1);
//     if (a == tmp)
//         tmp << 1, 0, 0;
//     b = (tmp - a * (a.transpose() * tmp)).normalized();
//     c = a.cross(b);
//     MatrixXd bc(3, 2);
//     bc.block<3, 1>(0, 0) = b;
//     bc.block<3, 1>(0, 1) = c;
//     return bc;
// }

// void Initializer::RefineGravity(VectorXd &x)
// {
//     Vector3d g0 = g_.normalized() * imu::g.norm();
//     Vector3d lx, ly;
//     int all_frame_count = frames_.size();
//     int n_state = all_frame_count * 3 + 2 + 1;

//     MatrixXd A{n_state, n_state};
//     A.setZero();
//     VectorXd b{n_state};
//     b.setZero();

//     Frames::iterator frame_i;
//     Frames::iterator frame_j;
//     for (int k = 0; k < 4; k++)
//     {
//         MatrixXd lxly(3, 2);
//         lxly = TangentBasis(g0);
//         int i = 0;
//         for (frame_i = frames_.begin(); next(frame_i) != frames_.end(); frame_i++, i++)
//         {
//             frame_j = next(frame_i);

//             MatrixXd tmp_A(6, 9);
//             tmp_A.setZero();
//             VectorXd tmp_b(6);
//             tmp_b.setZero();

//             double dt = frame_i->second->preintegration->sum_dt;

//             tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
//             tmp_A.block<3, 2>(0, 6) = frame_i->second->pose.rotationMatrix() * dt * dt / 2 * Matrix3d::Identity() * lxly;
//             tmp_A.block<3, 1>(0, 8) = frame_i->second->pose.rotationMatrix() * (frame_j->second->pose.inverse().translation() - frame_i->second->pose.inverse().translation()) / 100.0;
//             tmp_b.block<3, 1>(0, 0) = frame_i->second->preintegration->delta_p - frame_i->second->pose.rotationMatrix() * dt * dt / 2 * g0;

//             tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
//             tmp_A.block<3, 3>(3, 3) = frame_i->second->pose.rotationMatrix() * frame_j->second->pose.rotationMatrix().transpose();
//             tmp_A.block<3, 2>(3, 6) = frame_i->second->pose.rotationMatrix() * dt * Matrix3d::Identity() * lxly;
//             tmp_b.block<3, 1>(3, 0) = frame_i->second->preintegration->delta_v - frame_i->second->pose.rotationMatrix() * dt * Matrix3d::Identity() * g0;

//             // NOTE: remove useless cov_inv
//             MatrixXd r_A = tmp_A.transpose() * tmp_A;
//             VectorXd r_b = tmp_A.transpose() * tmp_b;

//             A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
//             b.segment<6>(i * 3) += r_b.head<6>();

//             A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
//             b.tail<3>() += r_b.tail<3>();

//             A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
//             A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
//         }
//         A = A * 1000.0;
//         b = b * 1000.0;
//         x = A.ldlt().solve(b);
//         VectorXd dg = x.segment<2>(n_state - 3);
//         g0 = (g0 + lxly * dg).normalized() * imu::g.norm();
//     }
//     g_ = g0;
// }

// bool Initializer::LinearAlignment(VectorXd &x)
// {
//     int all_frame_count = frames_.size();
//     int n_state = all_frame_count * 3 + 3 + 1;

//     MatrixXd A{n_state, n_state};
//     A.setZero();
//     VectorXd b{n_state};
//     b.setZero();

//     Frames::iterator frame_i;
//     Frames::iterator frame_j;
//     int i = 0;
//     for (frame_i = frames_.begin(); next(frame_i) != frames_.end(); frame_i++, i++)
//     {
//         frame_j = next(frame_i);

//         MatrixXd tmp_A(6, 10);
//         tmp_A.setZero();
//         VectorXd tmp_b(6);
//         tmp_b.setZero();

//         double dt = frame_i->second->preintegration->sum_dt;

//         tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
//         tmp_A.block<3, 3>(0, 6) = frame_i->second->pose.rotationMatrix() * dt * dt / 2 * Matrix3d::Identity();
//         tmp_A.block<3, 1>(0, 9) = frame_i->second->pose.rotationMatrix() * (frame_j->second->pose.inverse().translation() - frame_i->second->pose.inverse().translation()) / 100.0;
//         tmp_b.block<3, 1>(0, 0) = frame_i->second->preintegration->delta_p;
//         tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
//         tmp_A.block<3, 3>(3, 3) = frame_i->second->pose.rotationMatrix() * frame_j->second->pose.rotationMatrix().transpose();
//         tmp_A.block<3, 3>(3, 6) = frame_i->second->pose.rotationMatrix() * dt * Matrix3d::Identity();
//         tmp_b.block<3, 1>(3, 0) = frame_i->second->preintegration->delta_v;

//         // NOTE: remove useless con_inv
//         MatrixXd r_A = tmp_A.transpose() * tmp_A;
//         VectorXd r_b = tmp_A.transpose() * tmp_b;

//         A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
//         b.segment<6>(i * 3) += r_b.head<6>();

//         A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
//         b.tail<4>() += r_b.tail<4>();

//         A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
//         A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
//     }
//     A = A * 1000.0;
//     b = b * 1000.0;
//     x = A.ldlt().solve(b);
//     double s = x(n_state - 1) / 100.0; //scale
//     g_ = x.segment<3>(n_state - 4);    // g_
//     if (fabs(g_.norm() - imu::g.norm()) > 1.0 || s < 0)
//     {
//         return false;
//     }

//     RefineGravity(x);
//     s = (x.tail<1>())(0) / 100.0;
//     (x.tail<1>())(0) = s;
//     if (s < 0.0)
//         return false;
//     else
//         return true;
// }

// bool Initializer::VisualIMUAlignment(VectorXd &x)
// {
//     SolveGyroscopeBias();

//     if (LinearAlignment(x))
//         return true;
//     else
//         return false;
// }
} // namespace lvio_fusion