#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

namespace imu
{

//NOTE:translation,rotation,velocity,ba,bg,para_pose(rotation,translation)
int O_T = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12, O_PR = 0, O_PT = 4;
Vector3d g(0, 0, 9.81007);

Preintegration::Preintegration(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg)
    : linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
      jacobian{Matrix<double, 15, 15>::Identity()}, covariance{Matrix<double, 15, 15>::Zero()},
      sum_dt{0.0}, delta_p{Vector3d::Zero()}, delta_q{Quaterniond::Identity()}, delta_v{Vector3d::Zero()}
{
    delta_bias = Matrix<double, 6, 1>::Zero();
    noise = Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (Imu::Get()->ACC_N * Imu::Get()->ACC_N) * Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (Imu::Get()->GYR_N * Imu::Get()->GYR_N) * Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (Imu::Get()->ACC_N * Imu::Get()->ACC_N) * Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (Imu::Get()->GYR_N * Imu::Get()->GYR_N) * Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (Imu::Get()->ACC_W * Imu::Get()->ACC_W) * Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (Imu::Get()->GYR_W * Imu::Get()->GYR_W) * Matrix3d::Identity();
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

Matrix<double, 15, 1> Preintegration::Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                               const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj)
{
    //compute residuals
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
    residuals.block<3, 1>(O_T, 0) = Qi.inverse() * (0.5 * (Imu::Get()->Rwg * g) * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * ((Imu::Get()->Rwg * g) * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
}

Matrix<double, 15, 1> Preintegration::Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                               const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj, const Quaterniond &Rg)
{
    //compute residuals
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
    residuals.block<3, 1>(O_T, 0) = Qi.inverse() * (0.5 * (Rg * g) * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * ((Rg * g) * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
}

Vector3d Preintegration::GetUpdatedDeltaVelocity()
{
    Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);
    return delta_v + dv_dbg * delta_bias.block<3, 1>(0, 0) + dv_dba * delta_bias.block<3, 1>(3, 0);
}
Quaterniond Preintegration::GetUpdatedDeltaRotation()
{
    Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
    return delta_q * q_delta(dq_dbg * delta_bias.block<3, 1>(0, 0));
}
Vector3d Preintegration::GetUpdatedDeltaPosition()
{
    Matrix3d dp_dba = jacobian.block<3, 3>(O_T, O_BA);
    Matrix3d dp_dbg = jacobian.block<3, 3>(O_T, O_BG);
    return delta_p + dp_dbg * delta_bias.block<3, 1>(0, 0) + dp_dba * delta_bias.block<3, 1>(3, 0);
}
void Preintegration::SetNewBias(const Bias &new_bias)
{
    bias = new_bias;
    delta_bias(0) = new_bias.linearized_bg[0] - linearized_bg[0];
    delta_bias(1) = new_bias.linearized_bg[1] - linearized_bg[1];
    delta_bias(2) = new_bias.linearized_bg[2] - linearized_bg[2];
    delta_bias(3) = new_bias.linearized_ba[0] - linearized_ba[0];
    delta_bias(4) = new_bias.linearized_ba[1] - linearized_ba[1];
    delta_bias(5) = new_bias.linearized_ba[2] - linearized_ba[2];
}

Quaterniond Preintegration::GetDeltaRotation(const Bias &b_)
{
    Vector3d dbg;
    dbg << b_.linearized_bg[0] - linearized_bg[0], b_.linearized_bg[1] - linearized_bg[1], b_.linearized_bg[2] - linearized_bg[2];
    Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
    return delta_q * q_delta(dq_dbg * dbg);
}

Vector3d Preintegration::GetDeltaVelocity(const Bias &b_)
{
    Vector3d dbg;
    dbg << b_.linearized_bg[0] - linearized_bg[0], b_.linearized_bg[1] - linearized_bg[1], b_.linearized_bg[2] - linearized_bg[2];
    Vector3d dba;
    dba << b_.linearized_ba[0] - linearized_ba[0], b_.linearized_ba[1] - linearized_ba[1], b_.linearized_ba[2] - linearized_ba[2];
    Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);
    return delta_v + dv_dbg * dbg + dv_dba * dba;
}

Vector3d Preintegration::GetDeltaPosition(const Bias &b_)
{
    Vector3d dbg;
    dbg << b_.linearized_bg[0] - linearized_bg[0], b_.linearized_bg[1] - linearized_bg[1], b_.linearized_bg[2] - linearized_bg[2];
    Vector3d dba;
    dba << b_.linearized_ba[0] - linearized_ba[0], b_.linearized_ba[1] - linearized_ba[1], b_.linearized_ba[2] - linearized_ba[2];
    Matrix3d dp_dba = jacobian.block<3, 3>(O_T, O_BA);
    Matrix3d dp_dbg = jacobian.block<3, 3>(O_T, O_BG);
    return delta_p + dp_dbg * dbg + dp_dba * dba;
}

Bias Preintegration::GetDeltaBias(const Bias &b_)
{
    Vector3d dbg;
    dbg << b_.linearized_bg[0] - linearized_bg[0], b_.linearized_bg[1] - linearized_bg[1], b_.linearized_bg[2] - linearized_bg[2];
    Vector3d dba;
    dba << b_.linearized_ba[0] - linearized_ba[0], b_.linearized_ba[1] - linearized_ba[1], b_.linearized_ba[2] - linearized_ba[2];
    return Bias(dba, dbg);
}

} // namespace imu

} // namespace lvio_fusion