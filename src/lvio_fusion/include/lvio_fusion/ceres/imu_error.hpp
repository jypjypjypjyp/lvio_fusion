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


class InertialGSError
{
public:
    InertialGSError(imu::Preintegration::Ptr preintegration) : mpInt(preintegration) 
    {
        gl<< 0, 0, -9.81;
    }

    template <typename T>
    bool operator()(const T *Pose1,const T *V1,const T *Pose2, const T* V2,const T *GB,const T *AB,const T* rwg,const T* const scale, T *residuals) const
    {
        Quaternion<T> Qi(Pose1[3], Pose1[0], Pose1[1], Pose1[2]);
       Matrix<T, 3, 1> Pi(Pose1[4], Pose1[5], Pose1[6]);
        Matrix<T, 3, 1> Vi(V1[0], V1[1], V1[2]);
        Quaternion<T> Qj(Pose2[3], Pose2[0], Pose2[1], Pose2[2]);
        Matrix<T, 3, 1> Pj(Pose2[4], Pose2[5], Pose2[6]);
        Matrix<T, 3, 1> Vj(V2[0], V2[1], V2[2]);
        Matrix<T, 3, 1> gyroBias(GB[0], GB[1], GB[2]);
        Matrix<T, 3, 1> accBias(AB[0], AB[1], AB[2]);
        Matrix<T, 3, 1> eulerAngle(rwg[0], rwg[1], rwg[2]);
        T Scale=scale[0];
        T dt=T(mpInt->dT);
        Eigen::AngleAxis<T> rollAngle(AngleAxis<T>(eulerAngle(2),Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> pitchAngle(AngleAxis<T>(eulerAngle(1),Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> yawAngle(AngleAxis<T>(eulerAngle(0),Matrix<T, 3, 1>::UnitZ()));
        Matrix<T, 3, 3> Rwg;
        Rwg= yawAngle*pitchAngle*rollAngle;
        Matrix<T, 3, 1> g=Rwg*gl.cast<T>();
        const Bias  b1(double(accBias[0]),double(accBias[1]),double(accBias[2]),double(gyroBias[0]),double(gyroBias[1]),double(gyroBias[2]));
        const Matrix<T, 3, 3> dR = toMatrix3d(mpInt->GetDeltaRotation(b1)).cast<T>();
        const Matrix<T, 3, 1> dV = toVector3d(mpInt->GetDeltaVelocity(b1)).cast<T>();
        const Matrix<T, 3, 1> dP =toVector3d(mpInt->GetDeltaPosition(b1)).cast<T>();

        const Matrix<T, 3, 1> er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Matrix<T, 3, 1> ev = Qi.transpose()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Matrix<T, 3, 1> ep = Qi.transpose()*(Scale*(Pj - Pj - Vi*dt) - g*dt*dt/2) - dP;
        residuals[0]=er[0];
        residuals[1]=er[1];
        residuals[2]=er[2];
        residuals[3]=ev[0];
        residuals[4]=ev[1];
        residuals[5]=ev[2];
        residuals[6]=ep[0];
        residuals[7]=ep[1];
        residuals[8]=ep[2];
 return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration)
    {
        return (new ceres::AutoDiffCostFunction<InertialGSError,9, 7,3,7,3,3,3,3,1>(new InertialGSError(preintegration)));
    }
private:
    imu::Preintegration::Ptr mpInt;
    Vector3d gl;
};


class InertialError
{
public:
    InertialError(imu::Preintegration::Ptr preintegration) : mpInt(preintegration) {}
//P1,V1,g1,a1,P2,V2//7,3,3,3,7,3
    template <typename T>
    bool operator()(const T *P1, const T *V1, const T *g1, const T* a1,  const T*P2, const T* V2, T *residuals) const
    {
        Quaternion<T> Qi(P1[3], P1[0], P1[1], P1[2]);
        Matrix<T, 3, 1> Pi(P1[4], P1[5], P1[6]);
        Matrix<T, 3, 1> Vi(V1[0], V1[1], V1[2]);

        Matrix<T, 3, 1> gyroBias(g1[0], g1[1], g1[2]);
        Matrix<T, 3, 1> accBias(a1[0], a1[1],a1[2]);

        Quaternion<T> Qj(P2[3], P2[0], P2[1], P2[2]);
        Matrix<T, 3, 1> Pj(P2[4], P2[5], P2[6]);
        Matrix<T, 3, 1> Vj(V2[0], V2[1], V2[2]);
        T dt=T(mpInt->dT);
        Matrix<T, 3, 1> g;
        g<< 0, 0, -9.81;
        const Bias  b1(double(accBias[0]),double(accBias[1]),double(accBias[2]),double(gyroBias[0]),double(gyroBias[1]),double(gyroBias[2]));
        const Matrix<T, 3, 3> dR = toMatrix3d(mpInt->GetDeltaRotation(b1)).cast<T>();
        const Matrix<T, 3, 1> dV = toVector3d(mpInt->GetDeltaVelocity(b1)).cast<T>();
        const Matrix<T, 3, 1> dP =toVector3d(mpInt->GetDeltaPosition(b1)).cast<T>();

        const Matrix<T, 3, 1> er = LogSO3(dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Matrix<T, 3, 1> ev = Qi.transpose()*((Vj - Vi) - g*dt) - dV;
        const Matrix<T, 3, 1> ep = Qi.transpose()*((Pj - Pj - Vi*dt) - g*dt*dt/2) - dP;
        
        residuals[0]=er[0]
        residuals[1]=er[1];
        residuals[2]=er[2];
        residuals[3]=ev[0];
        residuals[4]=ev[1];
        residuals[5]=ev[2];
        residuals[6]=ep[0];
        residuals[7]=ep[1];
        residuals[8]=ep[2];
         return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration)
    {
        return (new ceres::AutoDiffCostFunction<InertialError,9, 7,3,3,3,7,3>(new InertialError(preintegration)));
    }
private:
    imu::Preintegration::Ptr mpInt;
    //Vector3d gl;
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
        return (new ceres::AutoDiffCostFunction<GyroRWError,3, 3,3,>(new GyroRWError()));
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
        return (new ceres::AutoDiffCostFunction<AccRWError,3, 3,3,>(new AccRWError()));
    }
};
/*
class StereoError
{
public:
    StereoError(){}

    template <typename T>
    bool operator()(T const *const *parameters, T *residuals) const
    {

        
    }
    static ceres::CostFunction *Create()
    {
        return (new ceres::AutoDiffCostFunction<StereoError,9, 3,7,>(new StereoError()));
    }
};
*/



// Converter
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}
Eigen::Matrix<double,3,3>  toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

} // namespace lvio_fusion

#endif //lvio_fusion_IMU_ERROR_H