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
        Vector3d acc, angVel;
        // if((i==0) && (i<(n-1)))
        // {
        //     double tab = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
        //     double tini = measureFromLastFrame[i].t- last_frame_time;
        //     acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a-
        //             (measureFromLastFrame[i+1].a-measureFromLastFrame[i].a)*(tini/tab))*0.5f;
        //     angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w-
        //             (measureFromLastFrame[i+1].w-measureFromLastFrame[i].w)*(tini/tab))*0.5f;
        //     tstep = measureFromLastFrame[i+1].t- last_frame_time;
        // }
        // else if(i<(n-1))
        // {
        //     acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a)*0.5f;
        //     angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w)*0.5f;
        //     tstep = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
        // }
        // else if((i>0) && (i==(n-1)))
        // {
        //     double tab = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
        //     double tend = measureFromLastFrame[i+1].t-current_frame_time;
        //     acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a-
        //             (measureFromLastFrame[i+1].a-measureFromLastFrame[i].a)*(tend/tab))*0.5f;
        //     angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w-
        //             (measureFromLastFrame[i+1].w-measureFromLastFrame[i].w)*(tend/tab))*0.5f;
        //     tstep = current_frame_time-measureFromLastFrame[i].t;
        // }
        // else if((i==0) && (i==(n-1)))
        // {
        //     acc = measureFromLastFrame[i].a;
        //     angVel = measureFromLastFrame[i].w;
        //     tstep = current_frame_time-last_frame_time;
        // }
        if((i==0) && (i==(n-1)))
        {
            acc = measureFromLastFrame[i].a;
            angVel = measureFromLastFrame[i].w;
            tstep = current_frame_time-last_frame_time;
        }
        else{
            acc = (measureFromLastFrame[i].a+measureFromLastFrame[i+1].a)*0.5f;
            angVel = (measureFromLastFrame[i].w+measureFromLastFrame[i+1].w)*0.5f;
            tstep = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
        }

       IntegrateNewMeasurement(acc,angVel,tstep);
    }
     isPreintegrated=true;
}


void Preintegration::IntegrateNewMeasurement(const Vector3d &acceleration, const Vector3d &angVel, const double &dt)
{
// 1.保存imu数据
 mvMeasurements.push_back(integrable(acceleration,angVel,dt));

 //Matrices to compute covariance
    Matrix<double,9,9> A = Matrix<double,9,9>::Identity();
    Matrix<double,9,6> B =MatrixXd::Zero(9,6);
    // 2.矫正加速度、角速度
    Matrix<double,3,1> acc ;
    acc<< acceleration[0]-b.linearized_ba[0],acceleration[1]-b.linearized_ba[1], acceleration[2]-b.linearized_ba[2];
    Matrix<double,3,1> accW;
    accW<< angVel[0]-b.linearized_bg[0], angVel[1]-b.linearized_bg[1], angVel[2]-b.linearized_bg[2];
    avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    avgW = (dT*avgW + accW*dt)/(dT+dt);

    // Update delta position dP and velocity dV (rely on no-updated delta rotation)
    // 3.更新dP，dV


    dP = dP + dV*dt + 0.5f*dR*acc*dt*dt;
    dV = dV + dR*acc*dt;


// 4.计算delta_x 的线性矩阵 eq.(62)
Matrix3d Wacc ;
Wacc << 0, -acc(2), acc(1),
                acc(2), 0, -acc(0),
                -acc(1), acc(0), 0;
    A.block<3,3>(3,0) = -dR*dt*Wacc;
    A.block<3,3>(6,0) = -0.5f*dR*dt*dt*Wacc;
    A.block<3,3>(6,3) = Matrix3d::Identity()*dt;
    B.block<3,3>(3,3)= dR*dt;
    B.block<3,3>(6,3) = 0.5f*dR*dt*dt;
// 5.更新bias雅克比 APPENDIX-A
    JPa = JPa + JVa*dt -0.5f*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5f*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;
// Update delta rotation
 //   cv::Point3f angVel_= cv::Point3f(angVel[0],angVel[1],angVel[2]);
  //  Vector3d angV=angVel;
    IntegratedRotation dRi( angVel,b,dt);
    dR = NormalizeRotation(dR*dRi.deltaR);

    // Compute rotation parts of matrices A and B
    A.block<3,3>(0,0) = dRi.deltaR.transpose();
    B.block<3,3>(0,0) = dRi.rightJ*dt;

    //*小量delta初始为0，更新后通常也为0，故省略了小量的更新

    // Update covariance
    // 6.更新协方差
    C.block<9,9>(0,0) = A*C.block<9,9>(0,0)*A.transpose() + B*Nga*B.transpose();
    C.block<6,6>(9,9) = C.block<6,6>(9,9) + NgaWalk;

    // Update rotation jacobian wrt bias correction
    JRg = dRi.deltaR.transpose()*JRg - dRi.rightJ*dt;

    // Total integrated time
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
    // R V P ba bg
    C =  Matrix<double,15,15>::Zero();
    // ba bg
    db =  Matrix<double,6,1>::Zero();
    b=b_;
    bu=b_;
    avgA = Vector3d::Zero();
    avgW = Vector3d::Zero();
    dT=0;
    mvMeasurements.clear();
}

Matrix<double,3,1> Preintegration::GetUpdatedDeltaVelocity()
{
    return dV + JVg*db.block<3,1>(0,0)+ JVa*db.block<3,1>(3,0);
}

void Preintegration::SetNewBias(const Bias &bu_)
{
    bu = bu_;

    db(0) = bu_.linearized_bg[0]-b.linearized_bg[0];
    db(1) = bu_.linearized_bg[1]-b.linearized_bg[1];
    db(2) = bu_.linearized_bg[2]-b.linearized_bg[2];
    db(3) = bu_.linearized_ba[0]-b.linearized_ba[0];
    db(4) = bu_.linearized_bg[1]-b.linearized_bg[1];
    db(5) = bu_.linearized_bg[2]-b.linearized_bg[2];
}

Matrix3d Preintegration::GetUpdatedDeltaRotation()
{
    return NormalizeRotation(dR*ExpSO3(JRg*db.block<3,1>(0,0)));
}
Matrix<double,3,1> Preintegration::GetUpdatedDeltaPosition()
{
    return dP + JPg*db.block<3,1>(0,0) + JPa*db.block<3,1>(3,0);
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
    return Bias(b_.linearized_ba[0]-b.linearized_ba[0],b_.linearized_ba[1]-b.linearized_ba[1],b_.linearized_ba[2]-b.linearized_ba[2] ,b_.linearized_bg[0]-b.linearized_bg[0],b_.linearized_bg[1]-b.linearized_bg[1],b_.linearized_bg[2]-b.linearized_bg[2]);
}



Matrix<double, 15, 1> Preintegration::Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                               const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj)
{
    Matrix<double, 15, 1> residuals;
    Matrix3d dp_dba=JPa;
    Matrix3d dp_dbg=JPg;
    Matrix3d dq_dbg=JRg;
    Matrix3d dv_dba=JVa;
    Matrix3d dv_dbg=JVg;
    Vector3d  linearized_ba(bu.linearized_ba[0],bu.linearized_ba[1],bu.linearized_ba[2]);
    Vector3d  linearized_bg(bu.linearized_bg[0],bu.linearized_bg[1],bu.linearized_bg[2]);
    Vector3d dba = Bai - linearized_ba;
    Vector3d dbg = Bgi - linearized_bg;
    Quaterniond delta_q;
    delta_q=dR;
    Vector3d delta_v=dV;
    Vector3d delta_p=dP;
    Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * dbg);
    Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
    residuals.block<3, 1>(O_T, 0) = Qi.inverse() * (0.5 * g * dT * dT + Pj - Pi - Vi * dT) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (g * dT + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
}
void Preintegration::Reintegrate()
{
    std::vector<integrable> aux = mvMeasurements;
    Initialize(bu);
    for(size_t i=0;i<aux.size();i++)
        IntegrateNewMeasurement(aux[i].a,aux[i].w,aux[i].t);
}

} // namespace imu

} // namespace lvio_fusion