#include "lvio_fusion/optimizer.h"


namespace lvio_fusion
{

//Converter
cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}


void optimizer::InertialOptimization(Map::Ptr pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel=false, bool bGauss=false, float priorG = 1e2, float priorA = 1e6)
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
     ceres::CostFunction *cost_function = PriorAccError::Create(cv::Mat::zeros(3,1,CV_32F));
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
               ceres::CostFunction *cost_function = InertialGSError::Create(current_frame->preintegration);
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
     Bias b(para_gyroBias[0],para_gyroBias[1],para_gyroBias[2],para_accBias[0],para_accBias[1],para_accBias[2]);
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
    
void optimizer::FullInertialBA(Map::Ptr pMap, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0, bool *pbStopFlag=NULL, bool bInit=false, float priorG = 1e2, float priorA=1e6, Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL)
{
auto KFs=pMap->GetAllKeyFrames();
auto MPs=pMap->GetAllLandmarks();
ceres::Problem problem;

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
     para_kfs[i]=para_kf;
     pIncKF=KFi;
     if(KFi->bImu)
     {
          auto para_v = KFi->mVw.data();
          para_vs[i]=para_v;
          if(!bInit)
          {
                Vector3d g;
               cv::cv2eigen(KFi->GetGyroBias(),g);
               auto para_gyroBias=g.data();
               para_gbs[i]=para_gyroBias;
               Vector3d a;
               cv::cv2eigen(KFi->GetAccBias(),a);
               auto para_accBias=a.data();
               para_abs[i]=para_accBias;
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
     Vector3d a;
     cv::cv2eigen(pIncKF->GetAccBias(),a);
     para_accBias=a.data();
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
          auto P1=para_kfs[i-1];
          auto V1=para_vs[i-1];
          double* g1,*g2,*a1,*a2;
          if(!bInit){
               g1=para_gbs[i-1];
               a1=para_abs[i-1];
               g2=para_gbs[i];
               a2=para_abs[i];
          }else
          {
               g1=para_gyroBias;
               a1=para_accBias;
          }
          auto P2=para_kfs[i];
          auto V2=para_vs[i];

          //ei p1,v1,g1,a1,p2,v2

          if(!bInit){
               //egr g1 g2
               //ear a1 a2  
          }
     }
     last_frame = current_frame;
}

//先验
if(bInit){
     //epa  para_accBias
     //epg para_gyroBias
}

//mappoint
   // const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
     
for(visual::Landmarks::iterator iter = MPs.begin(); iter != MPs.end(); iter++,ii++)
{
     visual::Landmark::Ptr mp=iter->second;
     auto para_point=mp->position.data();
     visual::Features observations = mp->observations;
     //bool bAllFixed = true;
     for(visual::Features::iterator iter=observations.begin();iter != observations.end(); iter++)
     {
          std::weak_ptr<Frame> KFi=iter->second->frame;
          //vp=para_kfi
          //e para_point para_kf

     }
     
}

//solve

//数据恢复


}

} // namespace lvio_fusion
