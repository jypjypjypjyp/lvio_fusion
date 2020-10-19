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

cv::Mat NormalizeRotation(const cv::Mat &R)
{
    cv::Mat U,w,Vt;
    cv::SVDecomp(R,w,U,Vt,cv::SVD::FULL_UV);
    // assert(cv::determinant(U*Vt)>0);
    return U*Vt;
}

void Preintegration::PreintegrateIMU(double last_frame_time,double current_frame_time)
{
    const int n = imuData_buf.size()-1;
     //IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Vector3d acc, angVel;
        // prev -> imuData_buf 的平均 acc，angVel
        if((i==0) && (i<(n-1)))
        {
            float tab = imuData_buf[i+1].t-imuData_buf[i].t;
            float tini = imuData_buf[i].t- last_frame_time;
            acc = (imuData_buf[i].a+imuData_buf[i+1].a-
                    (imuData_buf[i+1].a-imuData_buf[i].a)*(tini/tab))*0.5f;
            angVel = (imuData_buf[i].w+imuData_buf[i+1].w-
                    (imuData_buf[i+1].w-imuData_buf[i].w)*(tini/tab))*0.5f;
            tstep = imuData_buf[i+1].t- last_frame_time;
        }
        else if(i<(n-1))
        {
            acc = (imuData_buf[i].a+imuData_buf[i+1].a)*0.5f;
            angVel = (imuData_buf[i].w+imuData_buf[i+1].w)*0.5f;
            tstep = imuData_buf[i+1].t-imuData_buf[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = imuData_buf[i+1].t-imuData_buf[i].t;
            float tend = imuData_buf[i+1].t-current_frame_time;
            acc = (imuData_buf[i].a+imuData_buf[i+1].a-
                    (imuData_buf[i+1].a-imuData_buf[i].a)*(tend/tab))*0.5f;
            angVel = (imuData_buf[i].w+imuData_buf[i+1].w-
                    (imuData_buf[i+1].w-imuData_buf[i].w)*(tend/tab))*0.5f;
            tstep = current_frame_time-imuData_buf[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = imuData_buf[i].a;
            angVel = imuData_buf[i].w;
            tstep = current_frame_time-last_frame_time;
        }
        IntegrateNewMeasurement(acc,angVel,tstep);
        // 2.进行预积分
     //   mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
      //  pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }
    // 3.更新预积分状态
    //mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    // mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    //mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

   // mCurrentFrame.setIntegrated();

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
    IntegratedRotation dRi( angVel_,b,dt);
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
/*
void Preintegration::Repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg)
{
    sum_dt = 0.0;
    acc0 = linearized_acc;
    gyr0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
        Propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
}

void Preintegration::MidPointIntegration(double _dt,
                                         const Vector3d &_acc_0, const Vector3d &_gyr_0,
                                         const Vector3d &_acc_1, const Vector3d &_gyr_1,
                                         const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                                         const Vector3d &linearized_ba, const Vector3d &linearized_bg,
                                         Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
                                         Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian)
{
    Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
    Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
    Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    result_delta_v = delta_v + un_acc * _dt;
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    if (update_jacobian)
    {
        Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        Vector3d a_0_x = _acc_0 - linearized_ba;
        Vector3d a_1_x = _acc_1 - linearized_ba;
        Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x << 0, -w_x(2), w_x(1),
            w_x(2), 0, -w_x(0),
            -w_x(1), w_x(0), 0;
        R_a_0_x << 0, -a_0_x(2), a_0_x(1),
            a_0_x(2), 0, -a_0_x(0),
            -a_0_x(1), a_0_x(0), 0;
        R_a_1_x << 0, -a_1_x(2), a_1_x(1),
            a_1_x(2), 0, -a_1_x(0),
            -a_1_x(1), a_1_x(0), 0;

        MatrixXd F = MatrixXd::Zero(15, 15);
        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                              -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
        F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
        F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
        F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
        F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
        F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                              -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
        F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
        V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
}

void Preintegration::Propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1)
{
    dt = _dt;
    acc1 = _acc_1;
    gyr1 = _gyr_1;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    MidPointIntegration(_dt, acc0, gyr0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                        linearized_ba, linearized_bg,
                        result_delta_p, result_delta_q, result_delta_v,
                        result_linearized_ba, result_linearized_bg, true);

    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += dt;
    acc0 = acc1;
    gyr0 = gyr1;
}

Matrix<double, 15, 1> Preintegration::Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                               const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj)
{
    Matrix<double, 15, 1> residuals;
    Matrix3d dp_dba = jacobian.block<3, 3>(O_T, O_BA);
    Matrix3d dp_dbg = jacobian.block<3, 3>(O_T, O_BG);
    Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
    Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);
    Vector3d dba = Bai - linearized_ba;
    Vector3d dbg = Bgi - linearized_bg;
    Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * dbg);
    Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
    residuals.block<3, 1>(O_T, 0) = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (g * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
}
*/
} // namespace imu

} // namespace lvio_fusion