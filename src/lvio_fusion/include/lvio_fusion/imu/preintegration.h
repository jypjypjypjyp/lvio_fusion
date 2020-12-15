#ifndef lvio_fusion_PREINTEGRATION_H
#define lvio_fusion_PREINTEGRATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.hpp"

namespace lvio_fusion
{

class Frame;

namespace imu
{

extern int O_T, O_R, O_V, O_BA, O_BG, O_PR, O_PT;
extern Vector3d g;

class Preintegration
{
public:
    typedef std::shared_ptr<Preintegration> Ptr;

    static Preintegration::Ptr Create(Bias bias,const Imu::Ptr imu)
    {
        Preintegration::Ptr new_preintegration(new Preintegration(bias,imu));
        return new_preintegration;
    }
    Matrix<double, 15, 1> Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                   const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj,const Matrix3d Rwg);
    Matrix<double, 9, 1> Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                   const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj,const Matrix3d Rwg);

    void Appendimu(imuPoint imuMeasure)
    {
        imuData_buf.push_back(imuMeasure);
    }

    void PreintegrateIMU(std::vector<imuPoint> measureFromLastFrame,double last_frame_time,double current_frame_time);
   void IntegrateNewMeasurement(const Vector3d &acceleration, const Vector3d  &angVel, const double &dt);
    void Initialize(const Bias &b_);
    Matrix<double,3,1> GetUpdatedDeltaVelocity();
    void SetNewBias(const Bias &bu_);
    Matrix3d GetUpdatedDeltaRotation();
    Matrix<double,3,1> GetUpdatedDeltaPosition();
    Matrix3d GetDeltaRotation(const Bias &b_);
    Vector3d GetDeltaVelocity(const Bias &b_);
    Vector3d  GetDeltaPosition(const Bias &b_);
    Bias GetDeltaBias(const Bias &b_);
    void Reintegrate();

    // void MidPointIntegration(double _dt,
    //                          const Vector3d &_acc_0, const Vector3d &_gyr_0,
    //                          const Vector3d &_acc_1, const Vector3d &_gyr_1,
    //                          const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
    //                          const Vector3d &linearized_ba, const Vector3d &linearized_bg,
    //                          Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
    //                          Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian);

    // void Propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1);

   std::vector<imuPoint> imuData_buf;

    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf;
    std::vector<Vector3d> gyr_buf;



    double dT;
    Matrix<double,15,15> C;   //cov
    Matrix<double, 6, 6> Nga, NgaWalk;
    // double dt;
    // Vector3d acc0, gyr0;
    // Vector3d acc1, gyr1;
    // Matrix<double, 15, 15> jacobian, covariance;
    //Matrix<double, 18, 18> noise;

    // Values for the original bias (when integration was computed)
    Bias b;
    Matrix3d dR;
    Vector3d dV, dP;
    Matrix3d JRg, JVg, JVa, JPg, JPa; 
    Vector3d avgA;
    Vector3d avgW;
    
   // Updated bias
    Bias bu;
    // Dif between original and updated bias
    // This is used to compute the updated values of the preintegration
   Matrix<double,6,1> db;

   bool isPreintegrated;
private:
    Preintegration(){Initialize(Bias(0,0,0,0,0,0));};

    Preintegration(const Bias &b_,const Imu::Ptr imu)
{
    Nga.setZero();
    NgaWalk.setZero();
        Nga.block<3,3>(0,0)= (imu->GYR_N * imu->GYR_N) * Matrix3d::Identity();
        Nga.block<3,3>(3,3)= (imu->ACC_N * imu->ACC_N) * Matrix3d::Identity();
        NgaWalk.block<3,3>(0,0)=(imu->GYR_W * imu->GYR_W) * Matrix3d::Identity();
        NgaWalk.block<3,3>(3,3)= (imu->ACC_W * imu->ACC_W) * Matrix3d::Identity();
        Initialize(b_);
        isPreintegrated=false;
}

    struct integrable
    {
        integrable(const Vector3d &a_, const Vector3d &w_ , const double &t_):a(a_),w(w_),t(t_){}
        Vector3d a;
        Vector3d w;
        double t;
    };
    std::vector<integrable> mvMeasurements;


};

typedef std::map<double, Preintegration::Ptr> PreIntegrations;
} // namespace imu

} // namespace lvio_fusion

#endif // lvio_fusion_PREINTEGRATION_H