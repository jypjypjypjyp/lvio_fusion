#ifndef lvio_fusion_PREINTEGRATION_H
#define lvio_fusion_PREINTEGRATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.h"

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

    static Preintegration::Ptr Create(Bias bias)
    {
        Preintegration::Ptr new_preintegration(new Preintegration(bias));
        return new_preintegration;
    }

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

    Preintegration(const Bias &b_)
    {
        Nga.setZero();
        NgaWalk.setZero();
        Nga.block<3,3>(0,0)= (Imu::Get()->GYR_N * Imu::Get()->GYR_N) * Matrix3d::Identity();
        Nga.block<3,3>(3,3)= (Imu::Get()->ACC_N * Imu::Get()->ACC_N) * Matrix3d::Identity();
        NgaWalk.block<3,3>(0,0)=(Imu::Get()->GYR_W * Imu::Get()->GYR_W) * Matrix3d::Identity();
        NgaWalk.block<3,3>(3,3)= (Imu::Get()->ACC_W * Imu::Get()->ACC_W) * Matrix3d::Identity();
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
    std::vector<integrable> Measurements;

};

typedef std::map<double, Preintegration::Ptr> PreIntegrations;
} // namespace imu

} // namespace lvio_fusion

#endif // lvio_fusion_PREINTEGRATION_H