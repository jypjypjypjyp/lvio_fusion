#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/imu/imuOptimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{

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

void Initializer::ApplyScaledRotation(const Matrix3d &R,Frames keyframes)
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
        ImuOptimizer::InertialOptimization(keyframes,Rwg,1e1,1e4);
    }
    else{
        ImuOptimizer::InertialOptimization(keyframes,Rwg, priorG, priorA);
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
        ImuOptimizer::FullInertialBA(keyframes);
    }
    else{
        ImuOptimizer::FullInertialBA(keyframes, priorG, priorA);
    }

    bimu=true;
    initialized=true;
    reinit=false;
    return true;
}

} // namespace lvio_fusioni
