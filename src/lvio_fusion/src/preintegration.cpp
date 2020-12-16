#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{
namespace imu
{
//NOTE:translation,rotation,velocity,ba,bg,para_pose(rotation,translation)
int O_T = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12, O_PR = 0, O_PT = 4;
Vector3d g(0, 0, 9.8);

void Preintegration::PreintegrateIMU(std::vector<imuPoint> measureFromLastFrame,double last_frame_time,double current_frame_time)
{
    const int n = measureFromLastFrame.size()-1;
    for(int i=0; i<n; i++)
    {
        double tstep;
        double tab;
        Vector3d acc, angVel;
        if((i==0) && (i<(n-1)))
        {
             tab = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
            double tini = measureFromLastFrame[i].t- last_frame_time;
            acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a-
                    (measureFromLastFrame[i+1].a-measureFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w-
                    (measureFromLastFrame[i+1].w-measureFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = measureFromLastFrame[i+1].t- last_frame_time;
        }
        else if(i<(n-1))
        {
            acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a)*0.5f;
            angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w)*0.5f;
            tstep = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            tab = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
            double tend = measureFromLastFrame[i+1].t-current_frame_time;
            acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a-
                    (measureFromLastFrame[i+1].a-measureFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w-
                    (measureFromLastFrame[i+1].w-measureFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = current_frame_time-measureFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = measureFromLastFrame[i].a;
            angVel = measureFromLastFrame[i].w;
            tstep = current_frame_time-last_frame_time;
        }
         if(tab==0)continue;

       IntegrateNewMeasurement(acc,angVel,tstep);
    }
}


void Preintegration::IntegrateNewMeasurement(const Vector3d &acceleration, const Vector3d &angVel, const double &dt)
{
    if(!isPreintegrated) isPreintegrated=true;
    Measurements.push_back(integrable(acceleration,angVel,dt));

    Matrix<double,9,9> A = Matrix<double,9,9>::Identity();
    Matrix<double,9,6> B =MatrixXd::Zero(9,6);
    // 矫正加速度、角速度
    Matrix<double,3,1> acc ;
    acc<< acceleration[0]-b.linearized_ba[0],acceleration[1]-b.linearized_ba[1], acceleration[2]-b.linearized_ba[2];
    Matrix<double,3,1> accW;
    accW<< angVel[0]-b.linearized_bg[0], angVel[1]-b.linearized_bg[1], angVel[2]-b.linearized_bg[2];
    avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    avgW = (dT*avgW + accW*dt)/(dT+dt);

    dP = dP + dV*dt + 0.5f*dR*acc*dt*dt;
    dV = dV + dR*acc*dt;
    // 计算delta_x 的线性矩阵 eq.(62)
    Matrix3d Wacc ;
    Wacc << 0, -acc(2), acc(1),
                acc(2), 0, -acc(0),
                -acc(1), acc(0), 0;
    A.block<3,3>(3,0) = -dR*dt*Wacc;
    A.block<3,3>(6,0) = -0.5f*dR*dt*dt*Wacc;
    A.block<3,3>(6,3) = Matrix3d::Identity()*dt;
    B.block<3,3>(3,3)= dR*dt;
    B.block<3,3>(6,3) = 0.5f*dR*dt*dt;
    //更新bias雅克比 
    JPa = JPa + JVa*dt -0.5f*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5f*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;

    IntegratedRotation dRi( angVel,b,dt);
    dR = NormalizeRotation(dR*dRi.deltaR);

    A.block<3,3>(0,0) = dRi.deltaR.transpose();
    B.block<3,3>(0,0) = dRi.rightJ*dt;

    // 更新协方差
    C.block<9,9>(0,0) = A*C.block<9,9>(0,0)*A.transpose() + B*Nga*B.transpose();
    C.block<6,6>(9,9) = C.block<6,6>(9,9) + NgaWalk;
    JRg = dRi.deltaR.transpose()*JRg - dRi.rightJ*dt;

    dT += dt;
}
void Preintegration::Initialize(const Bias &b_)
{
    // R,V,P delta状态
    dR = Matrix3d::Identity();
    dV = Vector3d::Zero();
    dP = Vector3d::Zero();
    // R,V,P 分别对角速度，线加速度的雅克比
    JRg = Matrix3d::Zero();
    JVg = Matrix3d::Zero();
    JVa = Matrix3d::Zero();
    JPg = Matrix3d::Zero();
    JPa = Matrix3d::Zero();
    C =  Matrix<double,15,15>::Zero();
    delta_bias =  Matrix<double,6,1>::Zero();
    b=b_;
    bu=b_;
    avgA = Vector3d::Zero();
    avgW = Vector3d::Zero();
    dT=0;
    Measurements.clear();
}

Vector3d Preintegration::GetUpdatedDeltaVelocity()
{
    return dV + JVg*delta_bias.block<3,1>(0,0)+ JVa*delta_bias.block<3,1>(3,0);
}
Matrix3d Preintegration::GetUpdatedDeltaRotation()
{
    return NormalizeRotation(dR*ExpSO3(JRg*delta_bias.block<3,1>(0,0)));
}
Vector3d Preintegration::GetUpdatedDeltaPosition()
{
    return dP + JPg*delta_bias.block<3,1>(0,0) + JPa*delta_bias.block<3,1>(3,0);
}
void Preintegration::SetNewBias(const Bias &bu_)
{
    bu = bu_;

    delta_bias(0) = bu_.linearized_bg[0]-b.linearized_bg[0];
    delta_bias(1) = bu_.linearized_bg[1]-b.linearized_bg[1];
    delta_bias(2) = bu_.linearized_bg[2]-b.linearized_bg[2];
    delta_bias(3) = bu_.linearized_ba[0]-b.linearized_ba[0];
    delta_bias(4) = bu_.linearized_bg[1]-b.linearized_bg[1];
    delta_bias(5) = bu_.linearized_bg[2]-b.linearized_bg[2];
}

// 过去更新bias后的delta_R
Matrix3d Preintegration::GetDeltaRotation(const Bias &b_)
{
    Vector3d dbg;
    dbg << b_.linearized_bg[0]-b.linearized_bg[0],b_.linearized_bg[1]-b.linearized_bg[1],b_.linearized_bg[2]-b.linearized_bg[2];
    return NormalizeRotation(dR*ExpSO3(JRg*dbg));
}

Vector3d Preintegration::GetDeltaVelocity(const Bias &b_)
{

    Vector3d dbg ;
    dbg << b_.linearized_bg[0]-b.linearized_bg[0],b_.linearized_bg[1]-b.linearized_bg[1],b_.linearized_bg[2]-b.linearized_bg[2];
    Vector3d dba;
    dba << b_.linearized_ba[0]-b.linearized_ba[0],b_.linearized_ba[1]-b.linearized_ba[1],b_.linearized_ba[2]-b.linearized_ba[2];
    return dV + JVg*dbg + JVa*dba;
}

Vector3d Preintegration::GetDeltaPosition(const Bias &b_)
{
    Vector3d dbg ;
    dbg <<b_.linearized_bg[0]-b.linearized_bg[0],b_.linearized_bg[1]-b.linearized_bg[1],b_.linearized_bg[2]-b.linearized_bg[2];
    Vector3d dba;
    dba <<b_.linearized_ba[0]-b.linearized_ba[0],b_.linearized_ba[1]-b.linearized_ba[1],b_.linearized_ba[2]-b.linearized_ba[2];
    return dP + JPg*dbg + JPa*dba;
}

Bias Preintegration::GetDeltaBias(const Bias &b_)
{
    Vector3d dbg ;
    dbg <<b_.linearized_bg[0]-b.linearized_bg[0],b_.linearized_bg[1]-b.linearized_bg[1],b_.linearized_bg[2]-b.linearized_bg[2];
    Vector3d dba;
    dba <<b_.linearized_ba[0]-b.linearized_ba[0],b_.linearized_ba[1]-b.linearized_ba[1],b_.linearized_ba[2]-b.linearized_ba[2];
    return Bias(dba,dbg);
}

void Preintegration::Reintegrate()
{
    std::vector<integrable> aux = Measurements;
    Initialize(bu);
    for(size_t i=0;i<aux.size();i++)
        IntegrateNewMeasurement(aux[i].a,aux[i].w,aux[i].t);
}

} // namespace imu

} // namespace lvio_fusion