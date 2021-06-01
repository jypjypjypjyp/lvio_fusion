#ifndef lvio_fusion_IMU_ERROR_H
#define lvio_fusion_IMU_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

class ImuError : public ceres::SizedCostFunction<15, 7, 3, 3, 3, 7, 3, 3, 3>
{
public:
    ImuError(imu::Preintegration::Ptr preintegration) : preintegration_(preintegration) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Quaterniond Qi(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Vector3d Pi(parameters[0][4], parameters[0][5], parameters[0][6]);
        Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Vector3d Bai(parameters[2][0], parameters[2][1], parameters[2][2]);
        Vector3d Bgi(parameters[3][0], parameters[3][1], parameters[3][2]);

        Quaterniond Qj(parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]);
        Vector3d Pj(parameters[4][4], parameters[4][5], parameters[4][6]);
        Vector3d Vj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Vector3d Baj(parameters[6][0], parameters[6][1], parameters[6][2]);
        Vector3d Bgj(parameters[7][0], parameters[7][1], parameters[7][2]);
        Eigen::Map<Matrix<double, 15, 1>> residual(residuals);
        residual = preintegration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
        Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(preintegration_->covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;
        if (jacobians)
        {
            double sum_dt = preintegration_->sum_dt;
            Matrix3d dp_dba = preintegration_->jacobian.template block<3, 3>(imu::O_T, imu::O_BA);
            Matrix3d dp_dbg = preintegration_->jacobian.template block<3, 3>(imu::O_T, imu::O_BG);
            Matrix3d dq_dbg = preintegration_->jacobian.template block<3, 3>(imu::O_R, imu::O_BG);
            Matrix3d dv_dba = preintegration_->jacobian.template block<3, 3>(imu::O_V, imu::O_BA);
            Matrix3d dv_dbg = preintegration_->jacobian.template block<3, 3>(imu::O_V, imu::O_BG);

            if (jacobians[0])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PT) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PR) = skew_symmetric(Qi.inverse() * (0.5 * ( imu::g) * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Quaterniond corrected_delta_q = preintegration_->delta_q * q_delta(dq_dbg * (Bgi - preintegration_->linearized_bg));
                jacobian_pose_i.block<3, 3>(imu::O_R, imu::O_PR) = -(q_left(Qj.inverse() * Qi) * q_right(corrected_delta_q)).bottomRightCorner<3, 3>();
                jacobian_pose_i.block<3, 3>(imu::O_V, imu::O_PR) = skew_symmetric(Qi.inverse() * (( imu::g) * sum_dt + Vj - Vi));
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }
            if (jacobians[1])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(imu::O_T, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_v_i.block<3, 3>(imu::O_V, 0) = -Qi.inverse().toRotationMatrix();
                jacobian_v_i = sqrt_info * jacobian_v_i;
            }
            if (jacobians[2])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_ba_i(jacobians[2]);
                jacobian_ba_i.setZero();
                jacobian_ba_i.block<3, 3>(imu::O_T, 0) = -dp_dba;
                jacobian_ba_i.block<3, 3>(imu::O_V, 0) = -dv_dba;
                jacobian_ba_i.block<3, 3>(imu::O_BA, 0) = -Matrix3d::Identity();
                jacobian_ba_i = sqrt_info * jacobian_ba_i;
            }
            if (jacobians[3])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_bg_i(jacobians[3]);
                jacobian_bg_i.setZero();
                jacobian_bg_i.block<3, 3>(imu::O_T, 0) = -dp_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_R, 0) = -q_left(Qj.inverse() * Qi * preintegration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_V, 0) = -dv_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_BG, 0) = -Matrix3d::Identity();
                jacobian_bg_i = sqrt_info * jacobian_bg_i;
            }
            if (jacobians[4])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(imu::O_T, imu::O_PT) = Qi.inverse().toRotationMatrix();
                Quaterniond corrected_delta_q = preintegration_->delta_q * q_delta(dq_dbg * (Bgi - preintegration_->linearized_bg));
                jacobian_pose_j.block<3, 3>(imu::O_R, imu::O_PR) = q_left(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }
            if (jacobians[5])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_v_j(jacobians[5]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(imu::O_V, 0) = Qi.inverse().toRotationMatrix();
                jacobian_v_j = sqrt_info * jacobian_v_j;
            }
            if (jacobians[6])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_ba_j(jacobians[6]);
                jacobian_ba_j.setZero();
                jacobian_ba_j.block<3, 3>(imu::O_BA, 0) = Matrix3d::Identity();
                jacobian_ba_j = sqrt_info * jacobian_ba_j;
            }
            if (jacobians[7])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_bg_j(jacobians[7]);
                jacobian_bg_j.setZero();
                jacobian_bg_j.block<3, 3>(imu::O_BG, 0) = Matrix3d::Identity();
                jacobian_bg_j = sqrt_info * jacobian_bg_j;
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration)
    {
        return new ImuError(preintegration);
    }

private:
    imu::Preintegration::Ptr preintegration_;
};

class ImuInitError : public ceres::SizedCostFunction<15, 7, 3, 3, 3, 7, 3>
{
public:
    ImuInitError(imu::Preintegration::Ptr preintegration, double prior_a, double prior_g) : preintegration_(preintegration), prior_a_(prior_a), prior_g_(prior_g) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Quaterniond Qi(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Vector3d Pi(parameters[0][4], parameters[0][5], parameters[0][6]);

        Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Vector3d Bai(parameters[2][0], parameters[2][1], parameters[2][2]);
        Vector3d Bgi(parameters[3][0], parameters[3][1], parameters[3][2]);

        Quaterniond Qj(parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]);
        Vector3d Pj(parameters[4][4], parameters[4][5], parameters[4][6]);

        Vector3d Vj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Vector3d Baj(0, 0, 0);
        Vector3d Bgj(0, 0, 0);

        Eigen::Map<Matrix<double, 15, 1>> residual(residuals);
        residual = preintegration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
        Matrix<double, 15, 15> cov_inv = preintegration_->covariance.inverse();
        cov_inv.block<3, 3>(9, 9) = prior_a_ * Matrix3d::Identity();
        cov_inv.block<3, 3>(12, 12) = prior_g_ * Matrix3d::Identity();
        Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(cov_inv).matrixL().transpose();
        residual = sqrt_info * residual;
        // LOG(INFO) << residual;

        if (jacobians)
        {
            double sum_dt = preintegration_->sum_dt;
            Matrix3d dp_dba = preintegration_->jacobian.template block<3, 3>(imu::O_T, imu::O_BA);
            Matrix3d dp_dbg = preintegration_->jacobian.template block<3, 3>(imu::O_T, imu::O_BG);
            Matrix3d dq_dbg = preintegration_->jacobian.template block<3, 3>(imu::O_R, imu::O_BG);
            Matrix3d dv_dba = preintegration_->jacobian.template block<3, 3>(imu::O_V, imu::O_BA);
            Matrix3d dv_dbg = preintegration_->jacobian.template block<3, 3>(imu::O_V, imu::O_BG);

            if (jacobians[0])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PT) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PR) = skew_symmetric(Qi.inverse() * (0.5 * ( imu::g) * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Quaterniond corrected_delta_q = preintegration_->delta_q * q_delta(dq_dbg * (Bgi - preintegration_->linearized_bg));
                jacobian_pose_i.block<3, 3>(imu::O_R, imu::O_PR) = -(q_left(Qj.inverse() * Qi) * q_right(corrected_delta_q)).bottomRightCorner<3, 3>();
                jacobian_pose_i.block<3, 3>(imu::O_V, imu::O_PR) = skew_symmetric(Qi.inverse() * (( imu::g) * sum_dt + Vj - Vi));
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }
            if (jacobians[1])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(imu::O_T, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_v_i.block<3, 3>(imu::O_V, 0) = -Qi.inverse().toRotationMatrix();
                jacobian_v_i = sqrt_info * jacobian_v_i;
            }
            if (jacobians[2])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_ba_i(jacobians[2]);
                jacobian_ba_i.setZero();
                jacobian_ba_i.block<3, 3>(imu::O_T, 0) = -dp_dba;
                jacobian_ba_i.block<3, 3>(imu::O_V, 0) = -dv_dba;
                jacobian_ba_i.block<3, 3>(imu::O_BA, 0) = -Matrix3d::Identity();
                jacobian_ba_i = sqrt_info * jacobian_ba_i;
            }
            if (jacobians[3])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_bg_i(jacobians[3]);
                jacobian_bg_i.setZero();
                jacobian_bg_i.block<3, 3>(imu::O_T, 0) = -dp_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_R, 0) = -q_left(Qj.inverse() * Qi * preintegration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_V, 0) = -dv_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_BG, 0) = -Matrix3d::Identity();
                jacobian_bg_i = sqrt_info * jacobian_bg_i;
            }
            if (jacobians[4])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(imu::O_T, imu::O_PT) = Qi.inverse().toRotationMatrix();
                Quaterniond corrected_delta_q = preintegration_->delta_q * q_delta(dq_dbg * (Bgi - preintegration_->linearized_bg));
                jacobian_pose_j.block<3, 3>(imu::O_R, imu::O_PR) = q_left(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }
            if (jacobians[5])
            {
                Eigen::Map<Matrix<double, 15, 3, RowMajor>> jacobian_v_j(jacobians[5]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(imu::O_V, 0) = Qi.inverse().toRotationMatrix();
                jacobian_v_j = sqrt_info * jacobian_v_j;
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration, double prior_a, double prior_g)
    {
        return new ImuInitError(preintegration, prior_a, prior_g);
    }

private:
    imu::Preintegration::Ptr preintegration_;
    double prior_a_;
    double prior_g_;
};

class ImuInitGError
{
public:
    ImuInitGError(imu::Preintegration::Ptr preintegration, SE3d current_pose, SE3d last_pose, double prior_a, double prior_g) : preintegration_(preintegration), current_pose_(current_pose), last_pose_(last_pose), prior_a_(prior_a), prior_g_(prior_g) {}

    bool operator()(const double *_Vi, const double *_Bai, const double *_Bgi, const double *_Vj, const double *_Rg, double *residuals) const
    {
        Quaterniond Qi(last_pose_.rotationMatrix());
        Vector3d Pi = last_pose_.translation();
        Vector3d Vi(_Vi[0], _Vi[1], _Vi[2]);
        Vector3d Bai(_Bai[0], _Bai[1], _Bai[2]);
        Vector3d Bgi(_Bgi[0], _Bgi[1], _Bgi[2]);

        Quaterniond Qj(current_pose_.rotationMatrix());
        Vector3d Pj = current_pose_.translation();
        Vector3d Vj(_Vj[0], _Vj[1], _Vj[2]);
        Vector3d Baj(0, 0, 0);
        Vector3d Bgj(0, 0, 0);

        Quaterniond Rg(_Rg[3], _Rg[0], _Rg[1], _Rg[2]);

        Eigen::Map<Matrix<double, 15, 1>> residual(residuals);
        residual = preintegration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj, Rg);
        Matrix<double, 15, 15> cov_inv = preintegration_->covariance.inverse();
        cov_inv.block<3, 3>(9, 9) = prior_a_ * Matrix3d::Identity();
        cov_inv.block<3, 3>(12, 12) = prior_g_ * Matrix3d::Identity();
        Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(cov_inv).matrixL().transpose();
        residual = sqrt_info * residual;

        return true;
    }

    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration, SE3d current_pose, SE3d last_pose, double prior_a, double prior_g)
    {
        return new ceres::NumericDiffCostFunction<ImuInitGError, ceres::FORWARD, 15, 3, 3, 3, 3, 4>(new ImuInitGError(preintegration, current_pose, last_pose, prior_a, prior_g));
    }

private:
    imu::Preintegration::Ptr preintegration_;
    SE3d current_pose_;
    SE3d last_pose_;
    double prior_a_;
    double prior_g_;
};
} // namespace lvio_fusion

#endif //lvio_fusion_IMU_ERROR_H