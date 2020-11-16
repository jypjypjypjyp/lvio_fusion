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
        float tstep;
        Vector3d acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
            float tini = measureFromLastFrame[i].t- last_frame_time;
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
            float tab = measureFromLastFrame[i+1].t-measureFromLastFrame[i].t;
            float tend = measureFromLastFrame[i+1].t-current_frame_time;
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
       IntegrateNewMeasurement(acc,angVel,tstep);
    }
     isPreintegrated=true;
}


void Preintegration::IntegrateNewMeasurement(const Vector3d &acceleration, const Vector3d &angVel, const float &dt)
{
// 1.保存imu数据
 mvMeasurements.push_back(integrable(acceleration,angVel,dt));

 //Matrices to compute covariance
    cv::Mat A = cv::Mat::eye(9,9,CV_32F);
    cv::Mat B = cv::Mat::zeros(9,6,CV_32F);
    // 2.矫正加速度、角速度
    cv::Mat acc = (cv::Mat_<float>(3,1) << acceleration[0]-b.bax,acceleration[1]-b.bay, acceleration[2]-b.baz);
    cv::Mat accW = (cv::Mat_<float>(3,1) << angVel[0]-b.bwx, angVel[1]-b.bwy, angVel[2]-b.bwz);

    avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    avgW = (dT*avgW + accW*dt)/(dT+dt);

    // Update delta position dP and velocity dV (rely on no-updated delta rotation)
    // 3.更新dP，dV


    dP = dP + dV*dt + 0.5f*dR*acc*dt*dt;
    dV = dV + dR*acc*dt;


// 4.计算delta_x 的线性矩阵 eq.(62)
cv::Mat Wacc = (cv::Mat_<float>(3,3) << 0, -acc.at<float>(2), acc.at<float>(1),
                                                   acc.at<float>(2), 0, -acc.at<float>(0),
                                                   -acc.at<float>(1), acc.at<float>(0), 0);
    A.rowRange(3,6).colRange(0,3) = -dR*dt*Wacc;
    A.rowRange(6,9).colRange(0,3) = -0.5f*dR*dt*dt*Wacc;
    A.rowRange(6,9).colRange(3,6) = cv::Mat::eye(3,3,CV_32F)*dt;
    B.rowRange(3,6).colRange(3,6) = dR*dt;
    B.rowRange(6,9).colRange(3,6) = 0.5f*dR*dt*dt;
// 5.更新bias雅克比 APPENDIX-A
    JPa = JPa + JVa*dt -0.5f*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5f*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;
// Update delta rotation
    cv::Point3f angVel_= cv::Point3f(angVel[0],angVel[1],angVel[2]);
    Vector3d angV=angVel;
    IntegratedRotation dRi( angV,b,dt);
    dR = NormalizeRotation(dR*dRi.deltaR);

    // Compute rotation parts of matrices A and B
    A.rowRange(0,3).colRange(0,3) = dRi.deltaR.t();
    B.rowRange(0,3).colRange(0,3) = dRi.rightJ*dt;

    //*小量delta初始为0，更新后通常也为0，故省略了小量的更新

    // Update covariance
    // 6.更新协方差
    C.rowRange(0,9).colRange(0,9) = A*C.rowRange(0,9).colRange(0,9)*A.t() + B*Nga*B.t();
    C.rowRange(9,15).colRange(9,15) = C.rowRange(9,15).colRange(9,15) + NgaWalk;

    // Update rotation jacobian wrt bias correction
    JRg = dRi.deltaR.t()*JRg - dRi.rightJ*dt;

    // Total integrated time
    dT += dt;
}
void Preintegration::Initialize(const Bias &b_)
{
    // R,V,P delta状态
    dR = cv::Mat::eye(3,3,CV_32F);
    dV = cv::Mat::zeros(3,1,CV_32F);
    dP = cv::Mat::zeros(3,1,CV_32F);
    // R,V,P 分别对角速度，线加速度的雅克比
    JRg = cv::Mat::zeros(3,3,CV_32F);
    JVg = cv::Mat::zeros(3,3,CV_32F);
    JVa = cv::Mat::zeros(3,3,CV_32F);
    JPg = cv::Mat::zeros(3,3,CV_32F);
    JPa = cv::Mat::zeros(3,3,CV_32F);
    // R V P ba bg
    C = cv::Mat::zeros(15,15,CV_32F);
    Info=cv::Mat();
    // ba bg
    db = cv::Mat::zeros(6,1,CV_32F);
    b=b_;
    bu=b_;
    avgA = cv::Mat::zeros(3,1,CV_32F);
    avgW = cv::Mat::zeros(3,1,CV_32F);
    dT=0.0f;
    mvMeasurements.clear();
}

cv::Mat Preintegration::GetUpdatedDeltaVelocity()//TODO:db更新未写，用SetNewBias(const Bias &bu_)更新
{
    return dV + JVg*db.rowRange(0,3) + JVa*db.rowRange(3,6);
}

void Preintegration::SetNewBias(const Bias &bu_)
{
    bu = bu_;

    db.at<float>(0) = bu_.bwx-b.bwx;
    db.at<float>(1) = bu_.bwy-b.bwy;
    db.at<float>(2) = bu_.bwz-b.bwz;
    db.at<float>(3) = bu_.bax-b.bax;
    db.at<float>(4) = bu_.bay-b.bay;
    db.at<float>(5) = bu_.baz-b.baz;
}

cv::Mat Preintegration::GetUpdatedDeltaRotation()
{

    return NormalizeRotation(dR*ExpSO3(JRg*db.rowRange(0,3)));
}
cv::Mat Preintegration::GetUpdatedDeltaPosition()
{
    return dP + JPg*db.rowRange(0,3) + JPa*db.rowRange(3,6);
}


// 过去更新bias后的delta_R
cv::Mat Preintegration::GetDeltaRotation(const Bias &b_)
{

    cv::Mat dbg = (cv::Mat_<float>(3,1) << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    return NormalizeRotation(dR*ExpSO3(JRg*dbg));
}

cv::Mat Preintegration::GetDeltaVelocity(const Bias &b_)
{

    cv::Mat dbg = (cv::Mat_<float>(3,1) << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    cv::Mat dba = (cv::Mat_<float>(3,1) << b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz);
    return dV + JVg*dbg + JVa*dba;
}

cv::Mat Preintegration::GetDeltaPosition(const Bias &b_)
{
    cv::Mat dbg = (cv::Mat_<float>(3,1) << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    cv::Mat dba = (cv::Mat_<float>(3,1) << b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz);
    return dP + JPg*dbg + JPa*dba;
}

Bias Preintegration::GetDeltaBias(const Bias &b_)
{
    return Bias(b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz,b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
}



Matrix<double, 15, 1> Preintegration::Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                               const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj)
{
    Matrix<double, 15, 1> residuals;
    Matrix3d dp_dba;
    Matrix3d dp_dbg;
    Matrix3d dq_dbg;
    Matrix3d dv_dba;
    Matrix3d dv_dbg;
    cv::cv2eigen(JPa,dp_dba);
    cv::cv2eigen(JPg,dp_dbg);
    cv::cv2eigen(JRg,dq_dbg);
    cv::cv2eigen(JVa,dv_dba);
    cv::cv2eigen(JVg,dv_dbg);
    Vector3d  linearized_ba(bu.bax,bu.bay,bu.baz);
    Vector3d  linearized_bg(bu.bwx,bu.bwy,bu.bwz);
    Vector3d dba = Bai - linearized_ba;
    Vector3d dbg = Bgi - linearized_bg;
    Matrix3d dr;
    cv::cv2eigen(dR,dr);
    Quaterniond delta_q;
    delta_q=dr;
    Vector3d delta_v;
    cv::cv2eigen(dV,delta_v);
    Vector3d delta_p;
    cv::cv2eigen(dP,delta_p);
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
    const std::vector<integrable> aux = mvMeasurements;
    Initialize(bu);
    for(size_t i=0;i<aux.size();i++)
        IntegrateNewMeasurement(aux[i].a,aux[i].w,aux[i].t);
}

} // namespace imu

} // namespace lvio_fusion