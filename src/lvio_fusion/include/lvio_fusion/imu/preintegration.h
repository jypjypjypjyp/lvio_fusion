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
    void Appendimu(imuPoint imuMeasure)
    {
        imuData_buf.push_back(imuMeasure);
    }

    void PreintegrateIMU(std::vector<imuPoint> measureFromLastFrame,double last_frame_time,double current_frame_time);
    void IntegrateNewMeasurement(const Vector3d &acceleration, const Vector3d  &angVel, const double &dt);
    void Initialize(const Bias &b_);
    Vector3d GetUpdatedDeltaVelocity();
    void SetNewBias(const Bias &bu_);
    Matrix3d GetUpdatedDeltaRotation();
    Vector3d GetUpdatedDeltaPosition();
    Matrix3d GetDeltaRotation(const Bias &b_);
    Vector3d GetDeltaVelocity(const Bias &b_);
    Vector3d  GetDeltaPosition(const Bias &b_);
    Bias GetDeltaBias(const Bias &b_);
    void Reintegrate();

    std::vector<imuPoint> imuData_buf;
    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf;
    std::vector<Vector3d> gyr_buf;



    double dT;
    Matrix<double,15,15> C;   //cov
    Matrix<double, 6, 6> Nga, NgaWalk;
    Bias b;
    Matrix3d dR;
    Vector3d dV, dP;
    Matrix3d JRg, JVg, JVa, JPg, JPa; 
    Vector3d avgA;
    Vector3d avgW;
    Bias bu;
    Matrix<double,6,1> delta_bias;

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