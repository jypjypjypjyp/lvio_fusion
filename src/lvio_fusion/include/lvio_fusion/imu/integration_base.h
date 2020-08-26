#ifndef lvio_fusion_INTEGRATION_BASE_H
#define lvio_fusion_INTEGRATION_BASE_H

#include "lvio_fusion/imu/imu.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

namespace imu
{

class IntegrationBase
{
public:
    typedef std::shared_ptr<IntegrationBase> Ptr;

    static IntegrationBase::Ptr Create(const Vector3d &_acc_0, const Vector3d &_gyr_0, const Vector3d &_linearized_ba, const Vector3d &_linearized_bg, const Imu::Ptr imu)
    {
        IntegrationBase::Ptr new_preintegration(new IntegrationBase(_acc_0, _gyr_0, _linearized_ba, _linearized_bg, imu));
        return new_preintegration;
    }

    void push_back(double dt, const Vector3d &acc, const Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    void midPointIntegration(double _dt,
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
            //cout<<"A"<<endl<<A<<endl;

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

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }

    void propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;
    }

    Matrix<double, 15, 1> evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                   const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj)
    {
        static int O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12;
        Matrix<double, 15, 1> residuals;

        Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Vector3d dba = Bai - linearized_ba;
        Vector3d dbg = Bgi - linearized_bg;

        Quaterniond corrected_delta_q = delta_q * deltaQ(dq_dbg * dbg);
        Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (g * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt;
    Vector3d acc_0, gyr_0;
    Vector3d acc_1, gyr_1;
    const Vector3d linearized_acc, linearized_gyr;
    Vector3d linearized_ba, linearized_bg;
    Matrix<double, 15, 15> jacobian, covariance;
    Matrix<double, 15, 15> step_jacobian;
    Matrix<double, 15, 18> step_V;
    Matrix<double, 18, 18> noise;
    double sum_dt;
    Vector3d delta_p;
    Quaterniond delta_q;
    Vector3d delta_v;

    Vector3d Ba, Bg;
    std::vector<SE3d> poses;

    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf;
    std::vector<Vector3d> gyr_buf;

    Imu::Ptr imu_;

private:
    IntegrationBase(const Vector3d &_acc_0, const Vector3d &_gyr_0,
                    const Vector3d &_linearized_ba, const Vector3d &_linearized_bg, const Imu::Ptr imu)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
          jacobian{Matrix<double, 15, 15>::Identity()}, covariance{Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Vector3d::Zero()}, delta_q{Quaterniond::Identity()}, delta_v{Vector3d::Zero()}

    {
        noise = Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = (imu->ACC_N * imu->ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (imu->GYR_N * imu->GYR_N) * Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (imu->ACC_N * imu->ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (imu->GYR_N * imu->GYR_N) * Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (imu->ACC_W * imu->ACC_W) * Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (imu->GYR_W * imu->GYR_W) * Matrix3d::Identity();
    }
};
} // namespace imu

} // namespace lvio_fusion

#endif // lvio_fusion_INTEGRATION_BASE_H