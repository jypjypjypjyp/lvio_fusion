#ifndef lvio_fusion_IMU_ERROR_H
#define lvio_fusion_IMU_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/preintegration.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

const double G=9.81007;

class PriorGyroError:public ceres::SizedCostFunction<3, 3>
{
public:
    PriorGyroError(const Vector3d &bprior_,const double &priorG_):bprior(bprior_),priorG(priorG_){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
            Matrix3d info=priorG*Matrix3d::Identity();
            Matrix3d sqrt_info=LLT<Matrix3d>(info).matrixL().transpose();
            Vector3d gyroBias(parameters[0][0], parameters[0][1], parameters[0][2]);
            residuals[0]=(bprior[0]-gyroBias[0]);
            residuals[1]=(bprior[1]-gyroBias[1]);
            residuals[2]=(bprior[2]-gyroBias[2]);
               Eigen::Map<Vector3d> residual(residuals);
            residual = sqrt_info* residual;
            //LOG(INFO)<<" PriorGyroError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"    gyroBias  "<<gyroBias.transpose();
            if (jacobians)
            {
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg(jacobians[0]);
                jacobian_pg.setZero();
                jacobian_pg.block<3, 3>(0, 0) =- sqrt_info*Matrix3d::Identity();
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
          Matrix3d info=priorA*Matrix3d::Identity();
            Matrix3d sqrt_info=LLT<Matrix3d>(info).matrixL().transpose();
            Vector3d accBias(parameters[0][0], parameters[0][1], parameters[0][2]);
            residuals[0]=(bprior[0]-accBias[0]);
            residuals[1]=(bprior[1]-accBias[1]);
            residuals[2]=(bprior[2]-accBias[2]);
                           Eigen::Map<Vector3d> residual(residuals);
            residual = sqrt_info* residual;
            //LOG(INFO)<<" PriorAccError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"   accBias  "<<accBias.transpose();
            if (jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg(jacobians[0]);
                    jacobian_pg.setZero();
                    jacobian_pg.block<3, 3>(0, 0) =-sqrt_info*Matrix3d::Identity();

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

class InertialGSError:public ceres::SizedCostFunction<9, 3,3,3,3,4>
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
        Quaterniond rwg(parameters[4][3],parameters[4][0], parameters[4][1], parameters[4][2]);
        // double Scale=parameters[5][0];
        double Scale=1.0;
        double dt=(mpInt->dT);
        // Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(0),Vector3d::UnitX()));
        // Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1),Vector3d::UnitY()));
        // Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(2),Vector3d::UnitZ()));
        Matrix3d Rwg=rwg.toRotationMatrix();
        // Rwg= yawAngle*pitchAngle*rollAngle;
        Vector3d g=Rwg*gl;

        // LOG(INFO)<<"G  "<<(g).transpose()<<"Rwg"<<parameters4[0]<<" "<< parameters4[1]<<" "<< parameters4[2]<<" "<<parameters4[3];
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = (mpInt->GetDeltaRotation(b1));
        Vector3d dV = (mpInt->GetDeltaVelocity(b1));
        Vector3d dP =(mpInt->GetDeltaPosition(b1));

        const Vector3d er = LogSO3(dR.inverse()*Qi.inverse()*Qj);
        const Vector3d ev = Qi.inverse()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.inverse()*(Scale*(Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        
       Eigen::Map<Matrix<double, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
       //LOG(INFO)<<"InertialGSError residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
           Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
         Info = (Info+Info.transpose())/2;
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
         Matrix<double, 9,9> info_ = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
       Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>(info_).matrixL().transpose();
    
      //  assert(residual[0]<10&&residual[1]<10&&residual[2]<10&&residual[3]<10&&residual[4]<10&&residual[5]<10&&residual[6]<10&&residual[7]<10&&residual[8]<10);
        residual = sqrt_info* residual;
    //    LOG(INFO)<<"InertialGSError sqrt_info* residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
    //     LOG(INFO)<<"                Qi "<<Qi.eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
    //     LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
    //     LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
    //      LOG(INFO)<<"                 Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
        if (jacobians)
        {
            Bias delta_bias=mpInt->GetDeltaBias(b1);
            Vector3d dbg=delta_bias.linearized_bg;
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
            Eigen::Vector3d er_ = LogSO3(eR);
            Eigen::Matrix3d invJr = InverseRightJacobianSO3(er_);
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[0]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(3,0)=-s*Rbw1;
                jacobian_v_i.block<3, 3>(6,0)= -s*Rbw1*dt;
                jacobian_v_i=sqrt_info *jacobian_v_i;
            }
            if(jacobians[1])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[1]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(3,0)= s*Rbw1;
                jacobian_v_j=sqrt_info *jacobian_v_j;
            }
            if(jacobians[2])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_bg(jacobians[2]);
                jacobian_bg.setZero();
                jacobian_bg.block<3, 3>(0,0)= -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;
                jacobian_bg.block<3, 3>(3,0)=-JVg;
                jacobian_bg.block<3, 3>(6,0)=-JPg;
                jacobian_bg=sqrt_info *jacobian_bg;
            }
            if(jacobians[3])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_ba(jacobians[3]);
                jacobian_ba.setZero();
                jacobian_ba.block<3, 3>(3,0)=-JVa;
                jacobian_ba.block<3, 3>(6,0)=-JPa;
                jacobian_ba=sqrt_info *jacobian_ba;
            }
            if(jacobians[4])
            {
                Eigen::Map<Matrix<double, 9,4, RowMajor>> jacobian_Rwg(jacobians[4]);
                jacobian_Rwg.setZero();
                jacobian_Rwg.block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
                jacobian_Rwg.block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;
                jacobian_Rwg=sqrt_info *jacobian_Rwg;
            }
            //  if(jacobians[5]){
            //     Eigen::Map<Matrix<double, 9, 1>> jacobian_scale(jacobians[5]);
            //     jacobian_scale.setZero();
            //     jacobian_scale.block<3,1>(3,0) = Rbw1*(Vj-Vi);
            //     jacobian_scale.block<3,1>(6,0) = Rbw1*(Pj-Pi-Vi*dt);
            // }
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
         Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = mpInt->GetDeltaRotation(b1);
        Vector3d dV = mpInt->GetDeltaVelocity(b1);
        Vector3d dP =mpInt->GetDeltaPosition(b1);
      
        Vector3d er=  LogSO3(dR.inverse()*Qi.toRotationMatrix().inverse()*Qj.toRotationMatrix());
         Vector3d ev = Qi.inverse()*((Vj - Vi) - g*dt) - dV;
         Vector3d ep = Qi.inverse()*((Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        Eigen::Map<Matrix<double, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
        // LOG(INFO)<<"FullInertialBA residual "<<residual.transpose();
        //  // LOG(INFO)<<"\ndV "<<dV.transpose()<< "  dP "<<dP.transpose()<<"\ndR\n"<<dR;
        // LOG(INFO)<<"\nVj"<<(Vj).transpose()<<"Pj"<<(Pj).transpose()<<"\nQj\n"<<Qj.toRotationMatrix();
        Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
        Info = (Info+Info.transpose())/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
        Matrix<double, 9,9> info_ = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
        Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>( info_).matrixL().transpose();
        residual = sqrt_info* residual;
        if (jacobians)
        {
             Bias  b2(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
            Bias delta_bias=mpInt->GetDeltaBias(b2);
            Vector3d dbg=delta_bias.linearized_bg;
            Matrix3d Rwb1 =Qi.toRotationMatrix();
            //Matrix3d Rbw1 = Rwb1.transpose();
            Matrix3d Rbw1 = (Qi.inverse()).toRotationMatrix();
            Matrix3d Rwb2 = Qj.toRotationMatrix();

            Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
            Eigen::Vector3d er_ = LogSO3(eR);
            Eigen::Matrix3d invJr = InverseRightJacobianSO3(er_);
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(0,0)=  -invJr*Rwb2.transpose()*Rwb1;
                jacobian_pose_i.block<3, 3>(3,0)= Skew(Rbw1*((Vj - Vi)- g*dt));
                jacobian_pose_i.block<3, 3>(6,0)=  Skew(Rbw1*((Pj - Pi- Vi*dt)- 0.5*g*dt*dt));
                jacobian_pose_i.block<3, 3>(6,4)=  -Matrix3d::Identity();
                               LOG(INFO)<<jacobian_pose_i;
                 jacobian_pose_i=sqrt_info *jacobian_pose_i;
  
            }
            if(jacobians[1])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_i(jacobians[1]);
                jacobian_v_i.setZero();
                jacobian_v_i.block<3, 3>(3,0)=-Rbw1;
                jacobian_v_i.block<3, 3>(6,0)= -Rbw1*dt;
                jacobian_v_i=sqrt_info*jacobian_v_i;
            }
            if(jacobians[2])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_bg(jacobians[2]);
                jacobian_bg.setZero();
                jacobian_bg.block<3, 3>(0,0)= -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;
                jacobian_bg.block<3, 3>(3,0)=-JVg;
                jacobian_bg.block<3, 3>(6,0)=-JPg;
                jacobian_bg=sqrt_info*jacobian_bg;
            }
            if(jacobians[3])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_ba(jacobians[3]);
                jacobian_ba.setZero();
                jacobian_ba.block<3, 3>(3,0)=-JVa;
                jacobian_ba.block<3, 3>(6,0)=-JPa;
                jacobian_ba=sqrt_info*jacobian_ba;
            }
            if(jacobians[4])
            {
                Eigen::Map<Matrix<double, 9, 7, RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0,0)=invJr;
                jacobian_pose_j.block<3, 3>(6,4)=Rbw1*Rwb2;
                jacobian_pose_j=sqrt_info *jacobian_pose_j;
            }
            if(jacobians[5])
            {
                Eigen::Map<Matrix<double, 9, 3, RowMajor>> jacobian_v_j(jacobians[5]);
                jacobian_v_j.setZero();
                jacobian_v_j.block<3, 3>(3,0)= Rbw1;
                jacobian_v_j=sqrt_info *jacobian_v_j;
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

class GyroRWError :public ceres::SizedCostFunction<3, 3,3>
{
public:
    GyroRWError(const Matrix3d &priorG_):priorG(priorG_){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
             Matrix3d sqrt_info=LLT<Matrix3d>(priorG).matrixL().transpose();
            residuals[0]=parameters[1][0]-parameters[0][0];
            residuals[1]=parameters[1][1]-parameters[0][1];
            residuals[2]=parameters[1][2]-parameters[0][2];
            Eigen::Map<Matrix<double, 3, 1>>  residual(residuals);
            residual = sqrt_info * residual;
            //LOG(INFO)<<" GyroRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorG "<<priorG.block<1,1>(0,0);
            if (jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg1(jacobians[0]);
                    jacobian_pg1.setZero();
                    jacobian_pg1.block<3, 3>(0, 0) = -sqrt_info * Matrix3d::Identity();
                }
               if(jacobians[1])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pg2(jacobians[1]);
                    jacobian_pg2.setZero();
                    jacobian_pg2.block<3, 3>(0, 0) = sqrt_info * Matrix3d::Identity();
                }
            }
            return true;
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
          Matrix3d sqrt_info=LLT<Matrix3d>(priorA).matrixL().transpose();
            residuals[0]=parameters[1][0]-parameters[0][0];
            residuals[1]=parameters[1][1]-parameters[0][1];
            residuals[2]=parameters[1][2]-parameters[0][2];
            Eigen::Map<Matrix<double, 3, 1>>  residual(residuals);
             residual = sqrt_info * residual;
           //LOG(INFO)<<" AccRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorA "<<priorA.block<1,1>(0,0);
            if (jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pa1(jacobians[0]);
                    jacobian_pa1.setZero();
                    jacobian_pa1.block<3, 3>(0, 0) = - sqrt_info *Matrix3d::Identity();

                }
               if(jacobians[1])
                {
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>>  jacobian_pa2(jacobians[1]);
                    jacobian_pa2.setZero();
                    jacobian_pa2.block<3, 3>(0, 0) =  sqrt_info *Matrix3d::Identity();
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



class PriorGyroError2
{
public:
    PriorGyroError2(const Vector3d &bprior_,const double &priorG_):bprior(bprior_),priorG(priorG_){}
    bool operator()(const double*  parameters0, double* residuals) const 
    {
            Matrix3d info=priorG*Matrix3d::Identity();
            Matrix3d sqrt_info=LLT<Matrix3d>(info).matrixL().transpose();
            Vector3d gyroBias(parameters0[0], parameters0[1], parameters0[2]);
            residuals[0]=(bprior[0]-gyroBias[0]);
            residuals[1]=(bprior[1]-gyroBias[1]);
            residuals[2]=(bprior[2]-gyroBias[2]);
            Eigen::Map<Vector3d> residual(residuals);
            residual = sqrt_info* residual;

           // LOG(INFO)<<" PriorGyroError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"    gyroBias  "<<gyroBias.transpose();
            return true;

    }
    static ceres::CostFunction *Create(const Vector3d &v,const double &priorG_)
    {
        return new ceres::NumericDiffCostFunction<PriorGyroError2,ceres::FORWARD, 3,3>(new PriorGyroError2(v,priorG_));
    }
private:
    const Vector3d bprior;
    const double priorG;
};

class PriorAccError2
{
public:
    PriorAccError2(const Vector3d &bprior_,const double &priorA_):bprior(bprior_),priorA(priorA_){}
     bool operator()(const double*  parameters0, double* residuals) const 
    {
           Matrix3d info=priorA*Matrix3d::Identity();
            Matrix3d sqrt_info=LLT<Matrix3d>(info).matrixL().transpose();
            Vector3d accBias(parameters0[0], parameters0[1], parameters0[2]);
           residuals[0]=(bprior[0]-accBias[0]);
            residuals[1]=(bprior[1]-accBias[1]);
            residuals[2]=(bprior[2]-accBias[2]);
            Eigen::Map<Vector3d> residual(residuals);
            residual = sqrt_info* residual;
            //LOG(INFO)<<" PriorAccError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"   accBias  "<<accBias.transpose();
            return true;

    }
    static ceres::CostFunction *Create(const Vector3d &v,const double &priorA_)
    {
        return new ceres::NumericDiffCostFunction<PriorAccError2,ceres::FORWARD, 3,3>( new PriorAccError2(v,priorA_));
    }
private:
    const Vector3d bprior;
    const double priorA;
};

class InertialGSError2
{
public:
    InertialGSError2(imu::Preintegration::Ptr preintegration,SE3d current_pose_,SE3d last_pose_) : mpInt(preintegration),JRg(  preintegration->JRg),
    JVg( preintegration->JVg), JPg( preintegration->JPg), JVa( preintegration->JVa),
    JPa( preintegration->JPa),current_pose(current_pose_),last_pose(last_pose_)
    {
        gl<< 0, 0, -G;
    }
 bool operator()(const  double *  parameters0,const  double *  parameters1,const  double *  parameters2,const  double *  parameters3,const  double *  parameters4, double* residuals) const 
    {
    //    Quaterniond Qi(last_pose.rotationMatrix());
       Matrix3d Qi=last_pose.rotationMatrix();
        Vector3d Pi=last_pose.translation();
        Vector3d Vi(parameters0[0], parameters0[1], parameters0[2]);
       Matrix3d Qj=current_pose.rotationMatrix();
        // Quaterniond Qj(current_pose.rotationMatrix());
        Vector3d Pj=current_pose.translation();
        Vector3d Vj(parameters1[0], parameters1[1], parameters1[2]);
        Vector3d gyroBias(parameters2[0], parameters2[1], parameters2[2]);
        Vector3d accBias(parameters3[0], parameters3[1], parameters3[2]);
        Quaterniond rwg(parameters4[3],parameters4[0], parameters4[1], parameters4[2]);
        // double Scale=parameters[5][0];
        double Scale=1.0;
        double dt=(mpInt->dT);
        //Matrix3d Rwg=rwg.toRotationMatrix();
        Vector3d g=rwg*gl;
        // LOG(INFO)<<"G  "<<(g).transpose()<<"Rwg"<<parameters4[0]<<" "<< parameters4[1]<<" "<< parameters4[2]<<" "<<parameters4[3];
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = (mpInt->GetDeltaRotation(b1));
        Vector3d dV = (mpInt->GetDeltaVelocity(b1));
        Vector3d dP =(mpInt->GetDeltaPosition(b1));
      //  Quaterniond corrected_delta_q(dR);
       const Vector3d er=  LogSO3(dR.transpose()*Qi.transpose()*Qj);
        // const Vector3d er =  2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        const Vector3d ev = Qi.inverse()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.inverse()*(Scale*(Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        
       Eigen::Map<Matrix<double, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
       //LOG(INFO)<<"InertialGSError residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
           Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
         Info = (Info+Info.transpose())/2;
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
         Matrix<double, 9,9> info_ = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
        Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>( info_).matrixL().transpose();
        
      //  assert(residual[0]<10&&residual[1]<10&&residual[2]<10&&residual[3]<10&&residual[4]<10&&residual[5]<10&&residual[6]<10&&residual[7]<10&&residual[8]<10);
        residual = sqrt_info* residual;
      // LOG(INFO)<<"InertialGSError sqrt_info* residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
    //     LOG(INFO)<<"                Qi "<<Qi.eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
    //     LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
    //     LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
    //      LOG(INFO)<<"                 Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
        return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,SE3d current_pose_,SE3d last_pose_)
    {
        return new ceres::NumericDiffCostFunction<InertialGSError2,ceres::FORWARD,9, 3,3,3,3,4>(new InertialGSError2(preintegration,current_pose_,last_pose_));
    }
private:
 imu::Preintegration::Ptr mpInt;
    Vector3d gl;
    Matrix3d JRg, JVg, JPg;
    Matrix3d JVa, JPa;
    SE3d current_pose;
    SE3d last_pose;
};

class InertialError2
{
public:
    InertialError2(imu::Preintegration::Ptr preintegration,Matrix3d Rwg_) : mpInt(preintegration),JRg( preintegration->JRg),
    JVg(  preintegration->JVg), JPg( preintegration->JPg), JVa(  preintegration->JVa),
    JPa( preintegration->JPa) ,Rwg(Rwg_){}

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
        // g=Rwg*g;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = mpInt->GetDeltaRotation(b1);
        Vector3d dV = mpInt->GetDeltaVelocity(b1);
        Vector3d dP =mpInt->GetDeltaPosition(b1);
      
       const Vector3d er=  LogSO3(dR.inverse()*Qi.toRotationMatrix().inverse()*Qj.toRotationMatrix());
        // const Vector3d er =   2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        const Vector3d ev = Qi.inverse()*((Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.inverse()*((Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        Eigen::Map<Matrix<double, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
        // if(parameters2[0]==-0.0043304018657809229){
            LOG(INFO)<<"\n FullInertialBA residual "<<residual.transpose();
        // LOG(INFO)<<"\ndV "<<dV.transpose()<< "  dP "<<dP.transpose()<<"\ndR\n"<<dR;
        // LOG(INFO)<<"\nVj"<<(Vj).transpose()<<"Pj"<<(Pj).transpose()<<"\nQj\n"<<Qj.toRotationMatrix();
        LOG(INFO)<<"BA "<<accBias.transpose()<<" BG "<<gyroBias.transpose();  
        //}
           Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
         Info = (Info+Info.transpose())/2;
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
         Matrix<double, 9,9> info_ = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
        Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>( info_).matrixL().transpose();
        
        // LOG(INFO)<<"InertialError sqrt_info "<<sqrt_info;
        //assert(!isnan(residual[0])&&!isnan(residual[1])&&!isnan(residual[2])&&!isnan(residual[3])&&!isnan(residual[4])&&!isnan(residual[5])&&!isnan(residual[6])&&!isnan(residual[7])&&!isnan(residual[8]));
        residual = sqrt_info* residual;
        //LOG(INFO)<<"IMUError:  r "<<residual.transpose()<<"  "<<mpInt->dT;
        // LOG(INFO)<<"                Qi "<<Qi.toRotationMatrix().eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.toRotationMatrix().eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
        // LOG(INFO)<<"             Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
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
class GyroRWError2 
{
public:
    GyroRWError2(const Matrix3d &priorG_):priorG(priorG_){}

    bool operator()(const double*  parameters0, const double*  parameters1,double* residuals) const 
    {
            residuals[0]=parameters1[0]-parameters0[0];
            residuals[1]=parameters1[1]-parameters0[1];
            residuals[2]=parameters1[2]-parameters0[2];
            Eigen::Map<Matrix<double, 3, 1>>  residual(residuals);
            //assert(parameters1[0]<2&&parameters1[1]<2&&parameters1[2]<2&&parameters0[0]<2&&parameters0[1]<2&&parameters0[2]<2);
            residual = priorG* residual;
            //LOG(INFO)<<" GyroRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorG "<<priorG.block<1,1>(0,0);
            return true;
    }
    static ceres::CostFunction *Create(const Matrix3d &priorB_)
    {
        return new ceres::NumericDiffCostFunction<GyroRWError2,ceres::FORWARD, 3,3,3>(new GyroRWError2(priorB_));
    }
private:
    const Matrix3d priorG;
};
class AccRWError2
{
public:
    AccRWError2(const Matrix3d &priorA_):priorA(priorA_){}

    bool operator()(const double*  parameters0, const double*  parameters1,double* residuals) const 
    {
            residuals[0]=parameters1[0]-parameters0[0];
            residuals[1]=parameters1[1]-parameters0[1];
            residuals[2]=parameters1[2]-parameters0[2];
            Eigen::Map<Matrix<double, 3, 1>>  residual(residuals);
            //assert(parameters1[0]<2&&parameters1[1]<2&&parameters1[2]<2&&parameters0[0]<2&&parameters0[1]<2&&parameters0[2]<2);
            residual = priorA* residual;
            //LOG(INFO)<<" AccRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorA "<<priorA.block<1,1>(0,0);
            return true;
    }
    static ceres::CostFunction *Create(const Matrix3d &priorA_)
    {
        return new ceres::NumericDiffCostFunction<AccRWError2,ceres::FORWARD, 3,3,3>(new AccRWError2(priorA_));
    }
private:
    const Matrix3d priorA;
};

















class PriorGyroError3
{
public:
    PriorGyroError3(const Vector3d &bprior_,const double &priorG_):bprior(bprior_),priorG(priorG_){}
    template <typename T>
    bool operator()(const T*  parameters0, T* residuals) const 
    {
            Matrix<T, 3,3> info =Matrix<T,3,3>::Identity()*T(priorG);
            Matrix<T, 3,3> sqrt_info =LLT<Matrix<T, 3 ,3>>(info).matrixL().transpose();
            residuals[0]=(T(bprior[0])-parameters0[0]);
            residuals[1]=(T(bprior[1])-parameters0[1]);
            residuals[2]=(T(bprior[2])-parameters0[2]);
            Eigen::Map<Matrix<T, 3, 1>> residual(residuals);
            residual = sqrt_info* residual;
           // LOG(INFO)<<" PriorGyroError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"    gyroBias  "<<gyroBias.transpose();
            return true;

    }
    static ceres::CostFunction *Create(const Vector3d &v,const double &priorG_)
    {
        return new ceres::AutoDiffCostFunction<PriorGyroError3, 3,3>(new PriorGyroError3(v,priorG_));
    }
private:
    const Vector3d bprior;
    const double priorG;
};

class PriorAccError3
{
public:
    PriorAccError3(const Vector3d &bprior_,const double &priorA_):bprior(bprior_),priorA(priorA_){}
    template <typename T>
     bool operator()(const T*  parameters0, T* residuals) const 
    {
            Matrix<T, 3,3> info =Matrix<T,3,3>::Identity()*T(priorA);
            Matrix<T, 3,3> sqrt_info =LLT<Matrix<T, 3 ,3>>(info).matrixL().transpose();
            residuals[0]=(T(bprior[0])-parameters0[0]);
            residuals[1]=(T(bprior[1])-parameters0[1]);
            residuals[2]=(T(bprior[2])-parameters0[2]);
            Eigen::Map<Matrix<T, 3, 1>> residual(residuals);
            residual = sqrt_info* residual;
            //LOG(INFO)<<" PriorAccError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"   accBias  "<<accBias.transpose();
            return true;

    }
    static ceres::CostFunction *Create(const Vector3d &v,const double &priorA_)
    {
        return new ceres::AutoDiffCostFunction<PriorAccError3, 3,3>( new PriorAccError3(v,priorA_));
    }
private:
    const Vector3d bprior;
    const double priorA;
};

class InertialGSError3
{
public:
    InertialGSError3(imu::Preintegration::Ptr preintegration,SE3d current_pose_,SE3d last_pose_) : mpInt(preintegration),JRg(  preintegration->JRg),
    JVg( preintegration->JVg), JPg( preintegration->JPg), JVa( preintegration->JVa),
    JPa( preintegration->JPa),current_pose(current_pose_),last_pose(last_pose_)
    {
             Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
         Info = (Info+Info.transpose())/2;
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
         sqrt_info_ = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }
    template <typename T>
 bool operator()(const  T *  parameters0,const  T *  parameters1,const  T *  parameters2,const  T *  parameters3,const  T *  parameters4, T* residuals) const 
    {
        Matrix<T,3,3> Qi;
        Qi<<T(last_pose.rotationMatrix()(0,0)),T(last_pose.rotationMatrix()(0,1)),T(last_pose.rotationMatrix()(0,2)),
        T(last_pose.rotationMatrix()(1,0)),T(last_pose.rotationMatrix()(1,1)),T(last_pose.rotationMatrix()(1,2)),
        T(last_pose.rotationMatrix()(2,0)),T(last_pose.rotationMatrix()(2,1)),T(last_pose.rotationMatrix()(2,2));
        Matrix<T,3,1> Pi(T(last_pose.translation()(0,0)),T(last_pose.translation()(1,0)),T(last_pose.translation()(2,0)));
        Matrix<T,3,1> Vi(parameters0[0], parameters0[1], parameters0[2]);
        Matrix<T,3,3> Qj;
        Qj<<T(current_pose.rotationMatrix()(0,0)),T(current_pose.rotationMatrix()(0,1)),T(current_pose.rotationMatrix()(0,2)),
        T(current_pose.rotationMatrix()(1,0)),T(current_pose.rotationMatrix()(1,1)),T(current_pose.rotationMatrix()(1,2)),
        T(current_pose.rotationMatrix()(2,0)),T(current_pose.rotationMatrix()(2,1)),T(current_pose.rotationMatrix()(2,2));
          Matrix<T,3,1> Pj(T(current_pose.translation()(0,0)),T(current_pose.translation()(1,0)),T(current_pose.translation()(2,0)));
        Matrix<T,3,1> Vj(parameters1[0], parameters1[1], parameters1[2]);
        Quaternion<T> rwg(parameters4[3],parameters4[0], parameters4[1], parameters4[2]);
        // double Scale=parameters[5][0];
        T Scale=T(1.0);
        T dt=T(mpInt->dT);
        // Matrix<T,3,3>  Rwg=rwg.toRotationMatrix();
        Matrix<T,3,1> g( T(0), T(0), T(-G));
        g=rwg*g;
        //LOG(INFO)<<"g "<<(g).transpose();
        Matrix<T,3,1> dbg(parameters2[0]- T(mpInt->b.linearized_bg[0]),parameters2[1]- T(mpInt->b.linearized_bg[1]),parameters2[2]- T(mpInt->b.linearized_bg[2]));
        Matrix<T,3,1> dba(parameters3[0]- T(mpInt->b.linearized_ba[0]),parameters3[1]- T(mpInt->b.linearized_ba[1]),parameters3[2] - T(mpInt->b.linearized_ba[2]));
        
        Matrix<T,3,3> JRg_;
        JRg_<<T(JRg(0,0)),T(JRg(0,1)),T(JRg(0,2)),
        T(JRg(1,0)),T(JRg(1,1)),T(JRg(1,2)),
        T(JRg(2,0)),T(JRg(2,1)),T(JRg(2,2));
         Matrix<T,3,3>  JVg_;
         JVg_<<T(JVg(0,0)),T(JVg(0,1)),T(JVg(0,2)),
        T(JVg(1,0)),T(JVg(1,1)),T(JVg(1,2)),
        T(JVg(2,0)),T(JVg(2,1)),T(JVg(2,2));
          Matrix<T,3,3>  JVa_;
          JVa_<<T(JVa(0,0)),T(JVa(0,1)),T(JVa(0,2)),
        T(JVa(1,0)),T(JVa(1,1)),T(JVa(1,2)),
        T(JVa(2,0)),T(JVa(2,1)),T(JVa(2,2));
         Matrix<T,3,3>   JPg_;
        JPg_<<T(JPg(0,0)),T(JPg(0,1)),T(JPg(0,2)),
        T(JPg(1,0)),T(JPg(1,1)),T(JPg(1,2)),
        T(JPg(2,0)),T(JPg(2,1)),T(JPg(2,2));
            Matrix<T,3,3>   JPa_;
        JPa_<<T(JPa(0,0)),T(JPa(0,1)),T(JPa(0,2)),
        T(JPa(1,0)),T(JPa(1,1)),T(JPa(1,2)),
        T(JPa(2,0)),T(JPa(2,1)),T(JPa(2,2));
         Matrix<T,3,3>  dR_;
         dR_<<T(mpInt->dR(0,0)),T(mpInt->dR(0,1)),T(mpInt->dR(0,2)),
        T(mpInt->dR(1,0)),T(mpInt->dR(1,1)),T(mpInt->dR(1,2)),
        T(mpInt->dR(2,0)),T(mpInt->dR(2,1)),T(mpInt->dR(2,2));
         Matrix<T,3,1>  dV_(T(mpInt->dV(0,0)),T(mpInt->dV(1,0)),T(mpInt->dV(2,0)));
        Matrix<T,3,1>  dP_(T(mpInt->dP(0,0)),T(mpInt->dP(1,0)),T(mpInt->dP(2,0)));

        Matrix<T,3,1> JRg_dbg=(JRg_*dbg);
        Matrix<T,3,3> dR_ExpSO3_=(dR_*ExpSO3_(JRg_dbg));
        Quaternion<T> qdR_ExpSO3_(dR_ExpSO3_);
        Matrix<T,3,3> dR = qdR_ExpSO3_.toRotationMatrix();
        //Matrix<T,3,3> dR =NormalizeRotation_(dR_ExpSO3_);
        Matrix<T,3,1> dV = dV_ + JVg_*dbg + JVa_*dba;
        Matrix<T,3,1> dP =dP_ + JPg_*dbg + JPa_*dba;
      
      Matrix<T,3,3> eR= (dR.transpose()*Qi.transpose()*Qj);
        const Matrix<T,3,1> er =   LogSO3_(eR);
        const Matrix<T,3,1> ev = Qi.transpose()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Matrix<T,3,1> ep = Qi.transpose()*(Scale*(Pj - Pi - Vi*dt) - g*dt*dt/T(2)) - dP;
        
       Eigen::Map<Matrix<T, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
    //    LOG(INFO)<<"InertialGSError residual :  er "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"ev"<<residuals[3]<<" "<<residuals[4]<<" "<<residuals[5]<<"ep"<<residuals[6]<<" "<<residuals[7]<<" "<<residuals[8]<<" b "<<parameters2[0];
    //        Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
    //      Info = (Info+Info.transpose())/2;
    //      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    //      Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
    //      for(int i=0;i<9;i++)
    //          if(eigs[i]<1e-12)
    //              eigs[i]=0;
    //      Matrix<double, 9,9> sqrt_info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
      //       Matrix<T, 9,9> sqrt_info =LLT<Matrix<T, 9, 9>>( (mpInt->C.block<9,9>(0,0).inverse()).cast<T>()).matrixL().transpose();
    Matrix<T, 9,9> info =sqrt_info_.cast<T>();
    Matrix<T, 9,9> sqrt_info =LLT<Matrix<T, 9, 9>>(info).matrixL().transpose();


   //  LOG(INFO)<<sqrt_info;
        //    sqrt_info/=T(1000);
      //  assert(residual[0]<10&&residual[1]<10&&residual[2]<10&&residual[3]<10&&residual[4]<10&&residual[5]<10&&residual[6]<10&&residual[7]<10&&residual[8]<10);
        residual = sqrt_info* residual;
    //    LOG(INFO)<<"InertialGSError sqrt_info* residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
    //     LOG(INFO)<<"                Qi "<<Qi.eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
    //     LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
    //     LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
    //      LOG(INFO)<<"                 Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
        return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,SE3d current_pose_,SE3d last_pose_)
    {
        return (new ceres::AutoDiffCostFunction<InertialGSError3,9, 3,3,3,3,4>(new InertialGSError3(preintegration,current_pose_,last_pose_)));
    }
private:
 imu::Preintegration::Ptr mpInt;
    Matrix3d JRg, JVg, JPg;
    Matrix3d JVa, JPa;
    SE3d current_pose;
    SE3d last_pose;
        Matrix<double, 9,9> sqrt_info_;
};

class InertialError3
{
public:
    InertialError3(imu::Preintegration::Ptr preintegration) : mpInt(preintegration),JRg( preintegration->JRg),
    JVg(  preintegration->JVg), JPg( preintegration->JPg), JVa(  preintegration->JVa),
    JPa( preintegration->JPa){        
           Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
         Info = (Info+Info.transpose())/2;
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
         sqrt_info_ = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
        // LOG(INFO)<<"InertialError sqrt_info "<<sqrt_info_;

        }
template <typename T>
     bool operator()(const T*  parameters0, const T*  parameters1, const T*  parameters2, const T*  parameters3, const T*  parameters4, const T*  parameters5, T* residuals) const 
    {
        Quaternion<T> Qi(parameters0[3], parameters0[0], parameters0[1], parameters0[2]);
        Matrix<T,3,1> Pi(parameters0[4], parameters0[5], parameters0[6]);
        Matrix<T,3,1> Vi(parameters1[0], parameters1[1], parameters1[2]);

        Quaternion<T> Qj(parameters4[3], parameters4[0], parameters4[1], parameters4[2]);
        Matrix<T,3,1> Pj(parameters4[4], parameters4[5], parameters4[6]);
        Matrix<T,3,1> Vj(parameters5[0], parameters5[1], parameters5[2]);
        T dt=T(mpInt->dT);
        Matrix<T,3,1> g;
         g<< T(0), T(0), T(-G);
        // g=Rwg*g;

     Matrix<T,3,1> dbg(parameters2[0]- T(mpInt->b.linearized_bg[0]),parameters2[1]- T(mpInt->b.linearized_bg[1]),parameters2[2]- T(mpInt->b.linearized_bg[2]));
        Matrix<T,3,1> dba(parameters3[0]- T(mpInt->b.linearized_ba[0]),parameters3[1]- T(mpInt->b.linearized_ba[1]),parameters3[2] - T(mpInt->b.linearized_ba[2]));
        
        Matrix<T,3,3> JRg_;
        JRg_<<T(JRg(0,0)),T(JRg(0,1)),T(JRg(0,2)),
        T(JRg(1,0)),T(JRg(1,1)),T(JRg(1,2)),
        T(JRg(2,0)),T(JRg(2,1)),T(JRg(2,2));
         Matrix<T,3,3>  JVg_;
         JVg_<<T(JVg(0,0)),T(JVg(0,1)),T(JVg(0,2)),
        T(JVg(1,0)),T(JVg(1,1)),T(JVg(1,2)),
        T(JVg(2,0)),T(JVg(2,1)),T(JVg(2,2));
          Matrix<T,3,3>  JVa_;
          JVa_<<T(JVa(0,0)),T(JVa(0,1)),T(JVa(0,2)),
        T(JVa(1,0)),T(JVa(1,1)),T(JVa(1,2)),
        T(JVa(2,0)),T(JVa(2,1)),T(JVa(2,2));
         Matrix<T,3,3>   JPg_;
        JPg_<<T(JPg(0,0)),T(JPg(0,1)),T(JPg(0,2)),
        T(JPg(1,0)),T(JPg(1,1)),T(JPg(1,2)),
        T(JPg(2,0)),T(JPg(2,1)),T(JPg(2,2));
            Matrix<T,3,3>   JPa_;
        JPa_<<T(JPa(0,0)),T(JPa(0,1)),T(JPa(0,2)),
        T(JPa(1,0)),T(JPa(1,1)),T(JPa(1,2)),
        T(JPa(2,0)),T(JPa(2,1)),T(JPa(2,2));
         Matrix<T,3,3>  dR_;
         dR_<<T(mpInt->dR(0,0)),T(mpInt->dR(0,1)),T(mpInt->dR(0,2)),
        T(mpInt->dR(1,0)),T(mpInt->dR(1,1)),T(mpInt->dR(1,2)),
        T(mpInt->dR(2,0)),T(mpInt->dR(2,1)),T(mpInt->dR(2,2));
         Matrix<T,3,1>  dV_(T(mpInt->dV(0,0)),T(mpInt->dV(1,0)),T(mpInt->dV(2,0)));
        Matrix<T,3,1>  dP_(T(mpInt->dP(0,0)),T(mpInt->dP(1,0)),T(mpInt->dP(2,0)));

        Matrix<T,3,1> JRg_dbg=(JRg_*dbg);
        Matrix<T,3,3> dR_ExpSO3_=(dR_*ExpSO3_(JRg_dbg));
        Quaternion<T> qdR_ExpSO3_(dR_ExpSO3_);
        Matrix<T,3,3> dR = qdR_ExpSO3_.toRotationMatrix();
        //Matrix<T,3,3> dR =NormalizeRotation_(dR_ExpSO3_);
        Matrix<T,3,1> dV = dV_ + JVg_*dbg + JVa_*dba;
        Matrix<T,3,1> dP =dP_ + JPg_*dbg + JPa_*dba;
      

      Matrix<T,3,3> eR= (dR.transpose()*Qi.toRotationMatrix().transpose()*Qj.toRotationMatrix());
        const Matrix<T,3,1> er =   LogSO3_(eR);
        const Matrix<T,3,1> ev = Qi.inverse()*(Vj - Vi - g*dt) - dV;
        const Matrix<T,3,1> ep = Qi.inverse()*(Pj - Pi - Vi*dt - g*dt*dt/T(2)) - dP;
        Eigen::Map<Matrix<T, 9, 1>> residual(residuals);
        residual<<er,ev,ep;
// if(parameters2[0]==T(-0.0043304018657809229)){
//             LOG(INFO)<<"\n FullInertialBA residual "<<residual.transpose();
//         LOG(INFO)<<"\ndV "<<dV.transpose()<< "  dP "<<dP.transpose()<<"\ndR\n"<<dR;
//         LOG(INFO)<<"\nVj"<<(Vj).transpose()<<"Pj"<<(Pj).transpose()<<"\nQj\n"<<Qj.toRotationMatrix();
//         LOG(INFO)<<"BA "<<dba.transpose()<<" BG "<<dbg.transpose();  
// }
      // LOG(INFO)<<"eR:\n"<<eR;
        //  LOG(INFO)<<"InertialError residual: er "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<"ev"<<residuals[3]<<" "<<residuals[4]<<" "<<residuals[5]<<"ep"<<residuals[6]<<" "<<residuals[7]<<" "<<residuals[8]<<" dT "<<mpInt->dT;
        //    Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
        //  Info = (Info+Info.transpose())/2;
        //  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
        //  Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
        //  for(int i=0;i<9;i++)
        //      if(eigs[i]<1e-12)
        //          eigs[i]=0;
        //  Matrix<double, 9,9> sqrt_info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
       // Matrix<T, 9,9> sqrt_info =LLT<Matrix<T, 9, 9>>( (mpInt->C.block<9,9>(0,0).inverse()).cast<T>()).matrixL().transpose();
    Matrix<T, 9,9> info =sqrt_info_.cast<T>();
    Matrix<T, 9,9> sqrt_info =LLT<Matrix<T, 9, 9>>(info).matrixL().transpose();

        // sqrt_info/=T(1e6);
        //LOG(INFO)<<"InertialError sqrt_info "<<sqrt_info;
        //assert(!isnan(residual[0])&&!isnan(residual[1])&&!isnan(residual[2])&&!isnan(residual[3])&&!isnan(residual[4])&&!isnan(residual[5])&&!isnan(residual[6])&&!isnan(residual[7])&&!isnan(residual[8]));
        //assert(sqrt_info(0,0)<1e9);
        residual = sqrt_info* residual;

       // LOG(INFO)<<"IMUError:  r "<<residual.transpose()<<"  "<<mpInt->dT;
        // LOG(INFO)<<"                Qi "<<Qi.toRotationMatrix().eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.toRotationMatrix().eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
        // LOG(INFO)<<"             Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
         return true;
    }
    static ceres::CostFunction *Create(imu::Preintegration::Ptr preintegration,Matrix3d Rwg_)
    {
        return new ceres::AutoDiffCostFunction<InertialError3, 9, 7,3,3,3,7,3>(new InertialError3(preintegration));
    }
private:
    imu::Preintegration::Ptr mpInt;
    Matrix3d JRg, JVg, JPg;
    Matrix3d JVa, JPa;
    Matrix<double, 9,9> sqrt_info_;
};
class GyroRWError3
{
public:
    GyroRWError3(const Matrix3d &priorG_):priorG(priorG_){}
template <typename T>
    bool operator()(const T*  parameters0, const T*  parameters1,T* residuals) const 
    {
            Matrix<T, 3,3> info;
            info<<T(priorG(0,0)),T(priorG(0,1)),T(priorG(0,2)),
            T(priorG(1,0)),T(priorG(1,1)),T(priorG(1,2)),
            T(priorG(2,0)),T(priorG(2,1)),T(priorG(2,2));
            Matrix<T, 3,3> sqrt_info =LLT<Matrix<T, 3 ,3>>(info).matrixL().transpose();
            residuals[0]=(parameters1[0]-parameters0[0]);
            residuals[1]=(parameters1[1]-parameters0[1]);
            residuals[2]=(parameters1[2]-parameters0[2]);
            Eigen::Map<Matrix<T, 3, 1>> residual(residuals);
            residual = sqrt_info* residual;
            //LOG(INFO)<<" GyroRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorG "<<priorG.block<1,1>(0,0);
            return true;
    }
    static ceres::CostFunction *Create(const Matrix3d &priorB_)
    {
        return new ceres::AutoDiffCostFunction<GyroRWError3, 3,3,3>(new GyroRWError3(priorB_));
    }
private:
    const Matrix3d priorG;
};
class AccRWError3
{
public:
    AccRWError3(const Matrix3d &priorA_):priorA(priorA_){}
template <typename T>
    bool operator()(const T*  parameters0, const T*  parameters1,T* residuals) const
    { 
            Matrix<T, 3,3> info;
            info<<T(priorA(0,0)),T(priorA(0,1)),T(priorA(0,2)),
            T(priorA(1,0)),T(priorA(1,1)),T(priorA(1,2)),
            T(priorA(2,0)),T(priorA(2,1)),T(priorA(2,2));
            Matrix<T, 3,3> sqrt_info =LLT<Matrix<T, 3 ,3>>(info).matrixL().transpose();
            residuals[0]=(parameters1[0]-parameters0[0]);
            residuals[1]=(parameters1[1]-parameters0[1]);
            residuals[2]=(parameters1[2]-parameters0[2]);
            Eigen::Map<Matrix<T, 3, 1>> residual(residuals);
            residual = sqrt_info* residual;
            //LOG(INFO)<<" AccRWError: "<<residuals[0]<<" "<<residuals[1]<<" "<<residuals[2]<<" priorA "<<priorA.block<1,1>(0,0);
            return true;
    }
    static ceres::CostFunction *Create(const Matrix3d &priorA_)
    {
        return new ceres::AutoDiffCostFunction<AccRWError3, 3,3,3>(new AccRWError3(priorA_));
    }
private:
    const Matrix3d priorA;
};







} // namespace lvio_fusion
#endif //lvio_fusion_IMU_ERROR_H