#ifndef lvio_fusion_IMU_ERROR_H
#define lvio_fusion_IMU_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

const float GRAVITY_VALUE=9.81;

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
        Matrix<double, 15, 15> covariance;
        cv::cv2eigen(preintegration_->C,covariance);
        Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;
        LOG(INFO) << residual;
        
        if (jacobians)
        {
            double sum_dt = preintegration_->dT;
            Matrix3d dp_dba;
            Matrix3d dp_dbg;
            Matrix3d dq_dbg;
            Matrix3d dv_dba;
            Matrix3d dv_dbg;

            cv::cv2eigen(preintegration_->JPa,dp_dba);
            cv::cv2eigen(preintegration_->JPg,dp_dbg);
            cv::cv2eigen(preintegration_->JRg,dq_dbg);
            cv::cv2eigen(preintegration_->JVa,dv_dba);
            cv::cv2eigen(preintegration_->JVg,dv_dbg);

            Matrix3d dr;
            cv::cv2eigen(preintegration_->dR,dr);
            Quaterniond delta_q;
            delta_q=dr;
            Vector3d  linearized_bg(preintegration_->bu.bwx,preintegration_->bu.bwy,preintegration_->bu.bwz);
            
///int O_T = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12, O_PR = 0, O_PT = 4;
            if (jacobians[0])
            {
                Eigen::Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PT) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(imu::O_T, imu::O_PR) = skew_symmetric(Qi.inverse() * (0.5 * imu::g * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Quaterniond corrected_delta_q = delta_q * q_delta(dq_dbg * (Bgi - linearized_bg));
                jacobian_pose_i.block<3, 3>(imu::O_R, imu::O_PR) = -(q_left(Qj.inverse() * Qi) * q_right(corrected_delta_q)).bottomRightCorner<3, 3>();
                jacobian_pose_i.block<3, 3>(imu::O_V, imu::O_PR) = skew_symmetric(Qi.inverse() * (imu::g * sum_dt + Vj - Vi));
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

    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration)
    {
        return new ImuError(preintegration);
    }

private:
    imu::Preintegration::Ptr preintegration_;
};

class PriorGyroError:public ceres::SizedCostFunction<3, 3>
{
public:
    PriorGyroError(const Vector3d &bprior_):bprior(bprior_){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            Vector3d gyroBias(parameters[0][0], parameters[0][1], parameters[0][2]);
            residuals[0]=bprior[0]-gyroBias[0];
            residuals[1]=bprior[1]-gyroBias[1];
            residuals[2]=bprior[2]-gyroBias[2];
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg(jacobians[0]);
                jacobian_pg.setZero();
                jacobian_pg.block<3, 3>(0, 0) = Matrix3d::Identity();

            }
            return true;

    }
    static ceres::CostFunction *Create(const cv::Mat &bprior_)
    {
        Eigen::Matrix<double,3,1> v;
        v << bprior_.at<float>(0), bprior_.at<float>(1), bprior_.at<float>(2);
        return new PriorGyroError(v);
    }
private:
    const Vector3d bprior;
};

class PriorAccError:public ceres::SizedCostFunction<3, 3>
{
public:
    PriorAccError(const Vector3d &bprior_):bprior(bprior_){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            Vector3d accBias(parameters[0][0], parameters[0][1], parameters[0][2]);
            residuals[0]=bprior[0]-accBias[0];
            residuals[1]=bprior[1]-accBias[1];
            residuals[2]=bprior[2]-accBias[2];
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg(jacobians[0]);
                jacobian_pg.setZero();
                jacobian_pg.block<3, 3>(0, 0) = Matrix3d::Identity();

            }
            return true;

    }
    static ceres::CostFunction *Create(const cv::Mat &bprior_)
    {
        Eigen::Matrix<double,3,1> v;
        v << bprior_.at<float>(0), bprior_.at<float>(1), bprior_.at<float>(2);
        return new PriorAccError(v);
    }
private:
    const Vector3d bprior;
};


class InertialGSError:public ceres::SizedCostFunction<9, 7,3,7,3,3,3,3,1>
{
public:
    InertialGSError(imu::Preintegration::Ptr preintegration) : mpInt(preintegration),JRg( toMatrix3d( preintegration->JRg)),
    JVg( toMatrix3d( preintegration->JVg)), JPg( toMatrix3d( preintegration->JPg)), JVa( toMatrix3d( preintegration->JVa)),
    JPa( toMatrix3d( preintegration->JPa))
    {
        gl<< 0, 0, -GRAVITY_VALUE;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Quaterniond Qi(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Vector3d Pi(parameters[0][4], parameters[0][5], parameters[0][6]);
        Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Quaterniond Qj(parameters[2][3], parameters[2][0], parameters[2][1], parameters[2][2]);
        Vector3d Pj(parameters[2][4], parameters[2][5], parameters[2][6]);
        Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Vector3d gyroBias(parameters[4][0], parameters[4][1], parameters[4][2]);
        Vector3d accBias(parameters[5][0], parameters[5][1], parameters[5][2]);
        Vector3d eulerAngle(parameters[6][0], parameters[6][1], parameters[6][2]);
        double Scale=parameters[7][0];
        double dt=(mpInt->dT);
        Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(2),Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1),Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(0),Vector3d::UnitZ()));
        Matrix3d Rwg;
        Rwg= yawAngle*pitchAngle*rollAngle;
        Vector3d g=Rwg*gl;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        const Matrix3d dR = toMatrix3d(mpInt->GetDeltaRotation(b1));
        const Vector3d dV = toVector3d(mpInt->GetDeltaVelocity(b1));
        const Vector3d dP =toVector3d(mpInt->GetDeltaPosition(b1));

        const Vector3d er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Vector3d ev = Qi.toRotationMatrix().transpose()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.toRotationMatrix().transpose()*(Scale*(Pj - Pj - Vi*dt) - g*dt*dt/2) - dP;
        residuals[0]=er[0];
        residuals[1]=er[1];
        residuals[2]=er[2];
        residuals[3]=ev[0];
        residuals[4]=ev[1];
        residuals[5]=ev[2];
        residuals[6]=ep[0];
        residuals[7]=ep[1];
        residuals[8]=ep[2];
        if (jacobians)
        {
            Bias db=mpInt->GetDeltaBias(b1);
            Vector3d dbg;
            dbg << db.bwx, db.bwy, db.bwz;
            const Matrix3d Rwb1 =Qi.toRotationMatrix();
            const Matrix3d Rbw1 = Rwb1.transpose();
            const Matrix3d Rwb2 = Qj.toRotationMatrix();
            MatrixXd Gm = MatrixXd::Zero(3,2);
            Gm(0,1) = -GRAVITY_VALUE;
            Gm(1,0) = GRAVITY_VALUE;
            const double s = Scale;
            const Eigen::MatrixXd dGdTheta = Rwg*Gm;
            const Eigen::Matrix3d dR = toMatrix3d(mpInt->GetDeltaRotation(b1));
            const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
            const Eigen::Vector3d er = LogSO3(eR);
            const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);
            if(jacobians[0]){
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(0,0)=  -invJr*Rwb2.transpose()*Rwb1;
                jacobian_pose_i.block<3, 3>(3,0)= skew_symmetric(Rbw1*(s*(Vj - Vi)- g*dt));
                jacobian_pose_i.block<3, 3>(6,0)=  skew_symmetric(Rbw1*(s*(Pj - Pi- Vi*dt)- 0.5*g*dt*dt));
                jacobian_pose_i.block<3, 3>(6,4)=  -s*Matrix3d::Identity();
                
            }
            if(jacobians[1]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(3,0)=-s*Rbw1;
                jacobian_v_i.block<3, 3>(4,0)= -s*Rbw1*dt;
            }
            if(jacobians[2]){
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0,0)=invJr;
                jacobian_pose_j.block<3, 3>(6,3)=s*Rbw1*Rwb2;
                
            }
            if(jacobians[3]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[3]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(3,0)= s*Rbw1;
            }
            if(jacobians[4]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_bg(jacobians[4]);
                jacobian_bg.setZero();
                jacobian_bg.block<3, 3>(0,0)= -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;
                jacobian_bg.block<3, 3>(3,0)=-JVg;
                jacobian_bg.block<3, 3>(6,0)=-JPg;
            }
            if(jacobians[5]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_ba(jacobians[5]);
                jacobian_ba.setZero();
                jacobian_ba.block<3, 3>(3,0)=-JVa;
                jacobian_ba.block<3, 3>(6,0)=-JPa;
            }
            if(jacobians[6]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_Rwg(jacobians[6]);
                jacobian_Rwg.setZero();
                jacobian_Rwg.block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
                jacobian_Rwg.block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;
            }
            if(jacobians[7]){
                Eigen::Map<Matrix<double, 9, 1>> jacobian_scale(jacobians[7]);
                jacobian_scale.setZero();
                jacobian_scale.block<3,1>(3,0) = Rbw1*(Vj-Vi);
                jacobian_scale.block<3,1>(6,0) = Rbw1*(Pj-Pi-Vi*dt);

            }
        }
        return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration)
    {
        return new InertialGSError(preintegration);
    }
private:
    imu::Preintegration::Ptr mpInt;
    Vector3d gl;
    const Matrix3d JRg, JVg, JPg;
    const Matrix3d JVa, JPa;
};


class InertialError:public ceres::SizedCostFunction<9, 7,3,3,3,7,3>
{
public:
    InertialError(imu::Preintegration::Ptr preintegration) : mpInt(preintegration),JRg( toMatrix3d( preintegration->JRg)),
    JVg( toMatrix3d( preintegration->JVg)), JPg( toMatrix3d( preintegration->JPg)), JVa( toMatrix3d( preintegration->JVa)),
    JPa( toMatrix3d( preintegration->JPa)) {}
//P1,V1,g1,a1,P2,V2//7,3,3,3,7,3

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
        g<< 0, 0, -GRAVITY_VALUE;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        const Matrix3d dR = toMatrix3d(mpInt->GetDeltaRotation(b1));
        const Vector3d dV = toVector3d(mpInt->GetDeltaVelocity(b1));
        const Vector3d dP =toVector3d(mpInt->GetDeltaPosition(b1));

        const Vector3d er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Vector3d ev = Qi.toRotationMatrix().transpose()*((Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.toRotationMatrix().transpose()*((Pj - Pj - Vi*dt) - g*dt*dt/2) - dP;
        
        residuals[0]=er[0];
        residuals[1]=er[1];
        residuals[2]=er[2];
        residuals[3]=ev[0];
        residuals[4]=ev[1];
        residuals[5]=ev[2];
        residuals[6]=ep[0];
        residuals[7]=ep[1];
        residuals[8]=ep[2];

 if (jacobians)
        {
            Bias db=mpInt->GetDeltaBias(b1);
            Vector3d dbg;
            dbg << db.bwx, db.bwy, db.bwz;
            const Matrix3d Rwb1 =Qi.toRotationMatrix();
            const Matrix3d Rbw1 = Rwb1.transpose();
            const Matrix3d Rwb2 = Qj.toRotationMatrix();

            const Eigen::Matrix3d dR = toMatrix3d(mpInt->GetDeltaRotation(b1));
            const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
            const Eigen::Vector3d er = LogSO3(eR);
            const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);
            if(jacobians[0]){
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(0,0)=  -invJr*Rwb2.transpose()*Rwb1;
                jacobian_pose_i.block<3, 3>(3,0)= skew_symmetric(Rbw1*((Vj - Vi)- g*dt));
                jacobian_pose_i.block<3, 3>(6,0)=  skew_symmetric(Rbw1*((Pj - Pi- Vi*dt)- 0.5*g*dt*dt));
                jacobian_pose_i.block<3, 3>(6,4)=  -Matrix3d::Identity();
                
            }
            if(jacobians[1]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(3,0)=-Rbw1;
                jacobian_v_i.block<3, 3>(4,0)= -Rbw1*dt;
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
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0,0)=invJr;
                jacobian_pose_j.block<3, 3>(6,3)=Rbw1*Rwb2;
                
            }
            if(jacobians[5]){
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[5]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(3,0)= Rbw1;
            }
        }


         return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration)
    {
        return new InertialError(preintegration);
    }
private:
    imu::Preintegration::Ptr mpInt;
    //Vector3d gl;
    const Matrix3d JRg, JVg, JPg;
    const Matrix3d JVa, JPa;
};

class GyroRWError
{
public:
    GyroRWError(){}

    template <typename T>
    bool operator()(const T *g1,const T *g2, T *residuals) const
    {
            residuals[0]=g2[0]-g1[0];
            residuals[1]=g2[1]-g1[1];
            residuals[2]=g2[2]-g1[2];
             return true;
    }
    static ceres::CostFunction *Create()
    {
        return (new ceres::AutoDiffCostFunction<GyroRWError,3, 3,3>(new GyroRWError()));
    }
};
class AccRWError
{
public:
    AccRWError(){}

    template <typename T>
    bool operator()(const T *a1,const T *a2, T *residuals) const
    {
            residuals[0]=a2[0]-a1[0];
            residuals[1]=a2[1]-a1[1];
            residuals[2]=a2[2]-a1[2];
             return true;
    }
    static ceres::CostFunction *Create()
    {
        return (new ceres::AutoDiffCostFunction<AccRWError,3, 3,3>(new AccRWError()));
    }
};


} // namespace lvio_fusion

#endif //lvio_fusion_IMU_ERROR_H