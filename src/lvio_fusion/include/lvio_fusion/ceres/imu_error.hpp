#ifndef lvio_fusion_IMU_ERROR_H
#define lvio_fusion_IMU_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

const double G=9.81;

class ImuError : public ceres::SizedCostFunction<15, 7, 3, 3, 3, 7, 3, 3, 3>
{
public:
    ImuError(imu::Preintegration::Ptr preintegration,  Matrix3d Rwg_) : preintegration_(preintegration),Rwg(Rwg_) {}

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
        residual = preintegration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj,Rwg);
        Matrix<double, 15, 15> covariance=preintegration_->C;
        Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;
       // LOG(INFO)<<"IMUError:  r "<<residual.transpose()<<"  "<<preintegration_->dT;
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose();
        // LOG(INFO)<<"                 Bai "<< Bai.transpose()<<"  Bgi "<<  Bgi.transpose();
        // LOG(INFO)<<"                 Baj "<< Baj.transpose()<<"  Bgj "<<  Bgj.transpose();
        if (jacobians)
        {
            double sum_dt = preintegration_->dT;
            Matrix3d dp_dba=preintegration_->JPa;
            Matrix3d dp_dbg=preintegration_->JPg;
            Matrix3d dq_dbg=preintegration_->JRg;
            Matrix3d dv_dba=preintegration_->JVa;
            Matrix3d dv_dbg=preintegration_->JVg;
            Quaterniond delta_q;
            delta_q=preintegration_->dR;
            Vector3d  linearized_bg=preintegration_->bu.linearized_bg;
            
///int O_T = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12, O_PR = 0, O_PT = 4;
            if (jacobians[0])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PT) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PR) = skew_symmetric(Qi.inverse() * (0.5 *Rwg*imu::g * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * (Bgi - linearized_bg));
                jacobian_pose_i.block<3, 3>(imu::O_R, imu::O_PR) = -(q_left(Qj.inverse() * Qi) * q_right(corrected_delta_q)).bottomRightCorner<3, 3>();
                jacobian_pose_i.block<3, 3>(imu::O_V, imu::O_PR) = skew_symmetric(Qi.inverse() * (Rwg*imu::g * sum_dt + Vj - Vi));
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
                jacobian_bg_i.block<3, 3>(imu::O_R, 0) = -q_left(Qj.inverse() * Qi * delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_V, 0) = -dv_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_BG, 0) = -Matrix3d::Identity();
                jacobian_bg_i = sqrt_info * jacobian_bg_i;
            }
            if (jacobians[4])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(imu::O_T, imu::O_PT) = Qi.inverse().toRotationMatrix();
                Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * (Bgi -linearized_bg));
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

    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,  Matrix3d Rwg_)
    {
        return new ImuError(preintegration,Rwg_);
    }

private:
    Matrix3d Rwg;
    imu::Preintegration::Ptr preintegration_;
};

class ImuError2 : public ceres::SizedCostFunction<9, 7, 3, 3, 3, 7, 3>
{
public:
    ImuError2(imu::Preintegration::Ptr preintegration,  Matrix3d Rwg_) : preintegration_(preintegration),Rwg(Rwg_) {}

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
    
        Eigen::Map<Matrix<double, 9, 1>> residual(residuals);
        residual = preintegration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj,Rwg);
        Matrix<double, 15, 15> covariance=preintegration_->C;
        Matrix<double, 15, 15> sqrt_info_ = LLT<Matrix<double, 15, 15>>(covariance.inverse()).matrixL().transpose();
        Matrix<double, 9,9> sqrt_info =sqrt_info_.block<9,9>(0,0);
        residual = sqrt_info * residual;
        // LOG(INFO)<<"IMUError:  r "<<residual.transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose();
        // LOG(INFO)<<"                 Bai "<< Bai.transpose()<<"  Bgi "<<  Bgi.transpose();
        // LOG(INFO)<<"                 Baj "<< Baj.transpose()<<"  Bgj "<<  Bgj.transpose();
        if (jacobians)
        {
            double sum_dt = preintegration_->dT;
            Matrix3d dp_dba=preintegration_->JPa;
            Matrix3d dp_dbg=preintegration_->JPg;
            Matrix3d dq_dbg=preintegration_->JRg;
            Matrix3d dv_dba=preintegration_->JVa;
            Matrix3d dv_dbg=preintegration_->JVg;
            Quaterniond delta_q;
            delta_q=preintegration_->dR;
            Vector3d  linearized_bg=preintegration_->bu.linearized_bg;
            
///int O_T = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12, O_PR = 0, O_PT = 4;
            if (jacobians[0])
            {
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PT) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PR) = skew_symmetric(Qi.inverse() * (0.5 *Rwg*imu::g * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * (Bgi - linearized_bg));
                jacobian_pose_i.block<3, 3>(imu::O_R, imu::O_PR) = -(q_left(Qj.inverse() * Qi) * q_right(corrected_delta_q)).bottomRightCorner<3, 3>();
                jacobian_pose_i.block<3, 3>(imu::O_V, imu::O_PR) = skew_symmetric(Qi.inverse() * (Rwg*imu::g * sum_dt + Vj - Vi));
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }
            if (jacobians[1])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(imu::O_T, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_v_i.block<3, 3>(imu::O_V, 0) = -Qi.inverse().toRotationMatrix();
                jacobian_v_i = sqrt_info * jacobian_v_i;
            }
            if (jacobians[2])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_ba_i(jacobians[2]);
                jacobian_ba_i.setZero();
                jacobian_ba_i.block<3, 3>(imu::O_T, 0) = -dp_dba;
                jacobian_ba_i.block<3, 3>(imu::O_V, 0) = -dv_dba;
            }
            if (jacobians[3])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_bg_i(jacobians[3]);
                jacobian_bg_i.setZero();
                jacobian_bg_i.block<3, 3>(imu::O_T, 0) = -dp_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_R, 0) = -q_left(Qj.inverse() * Qi * delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_bg_i.block<3, 3>(imu::O_V, 0) = -dv_dbg;
            }
            if (jacobians[4])
            {
                Eigen::Map<Matrix<double,9, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(imu::O_T, imu::O_PT) = Qi.inverse().toRotationMatrix();
                Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * (Bgi -linearized_bg));
                jacobian_pose_j.block<3, 3>(imu::O_R, imu::O_PR) = q_left(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }
            if (jacobians[5])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[5]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(imu::O_V, 0) = Qi.inverse().toRotationMatrix();
                jacobian_v_j = sqrt_info * jacobian_v_j;
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,  Matrix3d Rwg_)
    {
        return new ImuError2(preintegration,Rwg_);
    }

private:
    Matrix3d Rwg;
    imu::Preintegration::Ptr preintegration_;
};

class PriorGyroError:public ceres::SizedCostFunction<3, 3>
{
public:
    PriorGyroError(const Vector3d &bprior_,const double &priorG_):bprior(bprior_),priorG(priorG_){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            Vector3d gyroBias(parameters[0][0], parameters[0][1], parameters[0][2]);
            residuals[0]=priorG*(bprior[0]-gyroBias[0]);
            residuals[1]=priorG*(bprior[1]-gyroBias[1]);
            residuals[2]=priorG*(bprior[2]-gyroBias[2]);
            //LOG(INFO)<<" PriorGyroError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2];
             if (jacobians)
            {
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg(jacobians[0]);
                jacobian_pg.setZero();
                jacobian_pg.block<3, 3>(0, 0) =- priorG*Matrix3d::Identity();
            }
            }
            return true;

    }
    static ceres::CostFunction *Create(const Vector3d &v,const double &priorG_)
    {
        return new PriorGyroError(v,priorG_);
    }
private:
    const Vector3d bprior;
    const double priorG;
};

class PriorAccError:public ceres::SizedCostFunction<3, 3>
{
public:
    PriorAccError(const Vector3d &bprior_,const double &priorA_):bprior(bprior_),priorA(priorA_){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            Vector3d accBias(parameters[0][0], parameters[0][1], parameters[0][2]);
            residuals[0]=priorA*(bprior[0]-accBias[0]);
            residuals[1]=priorA*(bprior[1]-accBias[1]);
            residuals[2]=priorA*(bprior[2]-accBias[2]);
            //LOG(INFO)<<" PriorAccError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2];
            if (jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg(jacobians[0]);
                    jacobian_pg.setZero();
                    jacobian_pg.block<3, 3>(0, 0) =-priorA*Matrix3d::Identity();

                }
            }
            return true;

    }
    static ceres::CostFunction *Create(const Vector3d &v,const double &priorA_)
    {
        return new PriorAccError(v,priorA_);
    }
private:
    const Vector3d bprior;
    const double priorA;
};

class InertialGSError:public ceres::SizedCostFunction<9, 3,3,3,3,3>
{
public:
    InertialGSError(imu::Preintegration::Ptr preintegration,SE3d current_pose_,SE3d last_pose_) : mpInt(preintegration),JRg(  preintegration->JRg),
    JVg( preintegration->JVg), JPg( preintegration->JPg), JVa( preintegration->JVa),
    JPa( preintegration->JPa),current_pose(current_pose_),last_pose(last_pose_)
    {
        gl<< 0, 0, -G;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Matrix3d Qi=last_pose.rotationMatrix();
        Vector3d Pi=last_pose.translation();
        Vector3d Vi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Matrix3d Qj=current_pose.rotationMatrix();
        Vector3d Pj=current_pose.translation();
        Vector3d Vj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Vector3d gyroBias(parameters[2][0], parameters[2][1], parameters[2][2]);
        Vector3d accBias(parameters[3][0], parameters[3][1], parameters[3][2]);
        Vector3d eulerAngle(parameters[4][0], parameters[4][1], parameters[4][2]);
        double Scale=1.0;
        double dt=(mpInt->dT);
        Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(0),Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1),Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(2),Vector3d::UnitZ()));
        Matrix3d Rwg;
        Rwg= yawAngle*pitchAngle*rollAngle;
        Vector3d g=Rwg*gl;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = (mpInt->GetDeltaRotation(b1));
        Vector3d dV = (mpInt->GetDeltaVelocity(b1));
        Vector3d dP =(mpInt->GetDeltaPosition(b1));

        const Vector3d er = LogSO3(dR.transpose()*Qi.transpose()*Qj);
        const Vector3d ev = Qi.transpose()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.transpose()*(Scale*(Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        residuals[0]=er[0];
        residuals[1]=er[1];
        residuals[2]=er[2];
        residuals[3]=ev[0];
        residuals[4]=ev[1];
        residuals[5]=ev[2];
        residuals[6]=ep[0];
        residuals[7]=ep[1];
        residuals[8]=ep[2];
     //LOG(INFO)<<"InertialGSError:  er "<<er.transpose()<<" ev "<<ev.transpose()<<" ep "<<ep.transpose();
        if (jacobians)
        {
            Bias db=mpInt->GetDeltaBias(b1);
            Vector3d dbg=db.linearized_bg;
            Matrix3d Rwb1 =Qi;
            Matrix3d Rbw1 = Rwb1.transpose();
            Matrix3d Rwb2 = Qj;
            MatrixXd Gm = MatrixXd::Zero(3,2);
            Gm(0,1) = -G;
            Gm(1,0) = G;
            double s = Scale;
            Eigen::MatrixXd dGdTheta = Rwg*Gm;
            Eigen::Matrix3d dR = (mpInt->GetDeltaRotation(b1));
            Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
            Eigen::Vector3d er = LogSO3(eR);
            Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);
            if(jacobians[0]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[0]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(3,0)=-s*Rbw1;
                jacobian_v_i.block<3, 3>(4,0)= -s*Rbw1*dt;
            }
            if(jacobians[1]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[1]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(3,0)= s*Rbw1;
            }
            if(jacobians[2]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_bg(jacobians[2]);
                jacobian_bg.setZero();
                jacobian_bg.block<3, 3>(0,0)= -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;
                jacobian_bg.block<3, 3>(3,0)=-JVg;
                jacobian_bg.block<3, 3>(6,0)=-JPg;
            }
            if(jacobians[3]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_ba(jacobians[3]);
                jacobian_ba.setZero();
                jacobian_ba.block<3, 3>(3,0)=-JVa;
                jacobian_ba.block<3, 3>(6,0)=-JPa;
            }
            if(jacobians[4]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_Rwg(jacobians[4]);
                jacobian_Rwg.setZero();
                jacobian_Rwg.block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
                jacobian_Rwg.block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;
            }
        }
        return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,SE3d current_pose_,SE3d last_pose_)
    {
        return new InertialGSError(preintegration,current_pose_,last_pose_);
    }
private:
    imu::Preintegration::Ptr mpInt;
    Vector3d gl;
    Matrix3d JRg, JVg, JPg;
    Matrix3d JVa, JPa;
    SE3d current_pose;
    SE3d last_pose;
};

class InertialError:public ceres::SizedCostFunction<9, 7,3,3,3,7,3>
{
public:
    InertialError(imu::Preintegration::Ptr preintegration,Matrix3d Rwg_) : mpInt(preintegration),JRg( preintegration->JRg),
    JVg(  preintegration->JVg), JPg( preintegration->JPg), JVa(  preintegration->JVa),
    JPa( preintegration->JPa) ,Rwg(Rwg_){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Quaterniond Qi(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Vector3d Pi(parameters[0][4], parameters[0][5], parameters[0][6]);
        Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        
        Vector3d gyroBias(parameters[2][0], parameters[2][1], parameters[2][2]);
        Vector3d accBias(parameters[3][0], parameters[3][1],parameters[3][2]);

        Quaterniond Qj(parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]);
        Vector3d Pj(parameters[4][4], parameters[4][5], parameters[4][6]);
        Vector3d Vj(parameters[5][0], parameters[5][1], parameters[5][2]);
        double dt=(mpInt->dT);
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = mpInt->GetDeltaRotation(b1);
        Vector3d dV = mpInt->GetDeltaVelocity(b1);
        Vector3d dP =mpInt->GetDeltaPosition(b1);
 
        const Vector3d er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Vector3d ev = Qi.toRotationMatrix().transpose()*((Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.toRotationMatrix().transpose()*((Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        
        residuals[0]=er[0];
        residuals[1]=er[1];
        residuals[2]=er[2];
        residuals[3]=ev[0];
        residuals[4]=ev[1];
        residuals[5]=ev[2];
        residuals[6]=ep[0];
        residuals[7]=ep[1];
        residuals[8]=ep[2];
        LOG(INFO)<<"InertialError:  er "<<er.transpose()<<" ev "<<ev.transpose()<<" ep "<<ep.transpose();
    //     LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose();
    //     LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose();
    //     LOG(INFO)<<"                dV "<<dV.transpose()<<" dP "<< dP.transpose();
    //    LOG(INFO) <<"                dR\n"<<dR.transpose();
    //     LOG(INFO)<<"                dt "<<dt;
         LOG(INFO)<<"        g "<<b1.linearized_bg.transpose()<<"        a "<<b1.linearized_ba.transpose();
 if (jacobians)
        {
            Bias db=mpInt->GetDeltaBias(b1);
            Vector3d dbg=db.linearized_bg;
            Matrix3d Rwb1 =Qi.toRotationMatrix();
            Matrix3d Rbw1 = Rwb1.transpose();
            Matrix3d Rwb2 = Qj.toRotationMatrix();

            Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1);
            Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
            Eigen::Vector3d er_ = LogSO3(eR);
            Eigen::Matrix3d invJr = InverseRightJacobianSO3(er_);
            if(jacobians[0]){
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(0,0)=  -invJr*Rwb2.transpose()*Rwb1;// er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
                jacobian_pose_i.block<3, 3>(3,0)= skew_symmetric(Rbw1*((Vj - Vi)- g*dt));//ev = Qi.toRotationMatrix().transpose()*((Vj - Vi) - g*dt) - dV;
                jacobian_pose_i.block<3, 3>(6,0)=  skew_symmetric(Rbw1*((Pj - Pi- Vi*dt)- 0.5*g*dt*dt));//ep
                jacobian_pose_i.block<3, 3>(6,4)=  -Matrix3d::Identity();
                 //LOG(INFO)<<"jacobians[0]\n"<<jacobian_pose_i;
            }
            if(jacobians[1]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(3,0)=-Rbw1;//ev
                jacobian_v_i.block<3, 3>(6,0)= -Rbw1*dt;//ep
                //LOG(INFO)<<"jacobians[1]\n"<<jacobian_v_i;
            }
            if(jacobians[2]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_bg(jacobians[2]);
                jacobian_bg.setZero();
                jacobian_bg.block<3, 3>(0,0)= -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;//er
                jacobian_bg.block<3, 3>(3,0)=-JVg;//ev
                jacobian_bg.block<3, 3>(6,0)=-JPg;//ep
                //LOG(INFO)<<"jacobians[2]\n"<<jacobian_bg;
            }
            if(jacobians[3]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_ba(jacobians[3]);
                jacobian_ba.setZero();
                jacobian_ba.block<3, 3>(3,0)=-JVa;
                jacobian_ba.block<3, 3>(6,0)=-JPa;
                //LOG(INFO)<<"jacobians[3]\n"<<jacobian_ba;
            }
            if(jacobians[4]){
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0,0)=invJr;//er
                jacobian_pose_j.block<3, 3>(6,4)=Rbw1*Rwb2;//ep
               // LOG(INFO)<<"jacobians[4]\n"<<jacobian_pose_j;
            }
            if(jacobians[5]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[5]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(3,0)= Rbw1;
               // LOG(INFO)<<"jacobians[5]\n"<<jacobian_v_j;
            }
        }


         return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,Matrix3d Rwg_)
    {
        return new InertialError(preintegration,Rwg_);
    }
private:
    imu::Preintegration::Ptr mpInt;
    Matrix3d Rwg;
    Matrix3d JRg, JVg, JPg;
    Matrix3d JVa, JPa;
};

class InertialError2
{
public:
    InertialError2(imu::Preintegration::Ptr preintegration,Matrix3d Rwg_) : mpInt(preintegration),JRg( preintegration->JRg),
    JVg(  preintegration->JVg), JPg( preintegration->JPg), JVa(  preintegration->JVa),
    JPa( preintegration->JPa) ,Rwg(Rwg_) {}

     bool operator()(const double*  parameters0, const double*  parameters1, const double*  parameters2, const double*  parameters3, const double*  parameters4, const double*  parameters5, double* residuals) const 
    {
        Quaterniond Qi(parameters0[3], parameters0[0], parameters0[1], parameters0[2]);
        Vector3d Pi(parameters0[4], parameters0[5], parameters0[6]);
        Vector3d Vi(parameters1[0], parameters1[1], parameters1[2]);
        
        Vector3d gyroBias(parameters2[0], parameters2[1], parameters2[2]);
        Vector3d accBias(parameters3[0], parameters3[1],parameters3[2]);

        Quaterniond Qj(parameters4[3], parameters4[0], parameters4[1], parameters4[2]);
        Vector3d Pj(parameters4[4], parameters4[5], parameters4[6]);
        Vector3d Vj(parameters5[0], parameters5[1], parameters5[2]);
        double dt=(mpInt->dT);
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = mpInt->GetDeltaRotation(b1);
        Vector3d dV = mpInt->GetDeltaVelocity(b1);
        Vector3d dP =mpInt->GetDeltaPosition(b1);

        const Vector3d er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Vector3d ev = Qi.toRotationMatrix().transpose()*((Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.toRotationMatrix().transpose()*((Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        
        // residuals[0]=er[0];
        // residuals[1]=er[1];
        // residuals[2]=er[2];
        // residuals[3]=ev[0];
        // residuals[4]=ev[1];
        // residuals[5]=ev[2];
        // residuals[6]=ep[0];
        // residuals[7]=ep[1];
        // residuals[8]=ep[2];
        Eigen::Map<Matrix<double, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
        //LOG(INFO)<<"InertialError2:  er "<<er.transpose()<<" ev "<<ev.transpose()<<" ep "<<ep.transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose();
        //  LOG(INFO)<<"                dV "<<dV.transpose()<<" dP "<< dP.transpose();
        //  LOG(INFO)<<"                dt "<<dt;
        // LOG(INFO)<<"                 g "<<b1.linearized_bg.transpose()<<"        a "<<b1.linearized_ba.transpose();
         return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,Matrix3d Rwg_)
    {
        return new ceres::NumericDiffCostFunction<InertialError2,ceres::FORWARD, 9, 7,3,3,3,7,3>(new InertialError2(preintegration,Rwg_));
    }
private:
    imu::Preintegration::Ptr mpInt;
    Matrix3d Rwg;
    Matrix3d JRg, JVg, JPg;
    Matrix3d JVa, JPa;
};


class GyroRWError :public ceres::SizedCostFunction<3, 3,3>
{
public:
    GyroRWError(const Matrix3d &priorG_):priorG(priorG_){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            residuals[0]=parameters[1][0]-parameters[0][0];
            residuals[1]=parameters[1][1]-parameters[0][1];
            residuals[2]=parameters[1][2]-parameters[0][2];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>>  residual(residuals);
            residual = priorG * residual;
           // LOG(INFO)<<" GyroRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorG "<<priorG.block<1,1>(0,0);
            if (jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg1(jacobians[0]);
                    jacobian_pg1.setZero();
                    jacobian_pg1.block<3, 3>(0, 0) = -priorG * Matrix3d::Identity();
                }
               if(jacobians[1])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg2(jacobians[1]);
                    jacobian_pg2.setZero();
                    jacobian_pg2.block<3, 3>(0, 0) = priorG * Matrix3d::Identity();
                }
            }
    }
    static ceres::CostFunction *Create(const Matrix3d &priorB_)
    {
        return new GyroRWError(priorB_);
    }
private:
    const Matrix3d priorG;
};
class AccRWError:public ceres::SizedCostFunction<3, 3,3>
{
public:
    AccRWError(const Matrix3d &priorA_):priorA(priorA_){}

   virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            residuals[0]=parameters[1][0]-parameters[0][0];
            residuals[1]=parameters[1][1]-parameters[0][1];
            residuals[2]=parameters[1][2]-parameters[0][2];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>>  residual(residuals);
             residual = priorA * residual;
           // LOG(INFO)<<" AccRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorA "<<priorA.block<1,1>(0,0);
            if (jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pa1(jacobians[0]);
                    jacobian_pa1.setZero();
                    jacobian_pa1.block<3, 3>(0, 0) = - priorA *Matrix3d::Identity();
                }
               if(jacobians[1])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pa2(jacobians[1]);
                    jacobian_pa2.setZero();
                    jacobian_pa2.block<3, 3>(0, 0) =  priorA *Matrix3d::Identity();
                }
            }
            return true;
    }
    static ceres::CostFunction *Create(const Matrix3d &priorA_)
    {
        return new AccRWError(priorA_);
    }
private:
    const Matrix3d priorA;
};


} // namespace lvio_fusion

#endif //lvio_fusion_IMU_ERROR_H