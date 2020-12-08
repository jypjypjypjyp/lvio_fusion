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

    static Preintegration::Ptr Create(Bias bias, Calib ImuCalib_,const Imu::Ptr imu)
    {
        Preintegration::Ptr new_preintegration(new Preintegration(bias, ImuCalib_,imu));
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

   std::vector<imuPoint> imuData_buf;

    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf;
    std::vector<Vector3d> gyr_buf;



    double dT;
    Matrix<double,15,15> C;   //cov
    Matrix<double, 6, 6> Nga, NgaWalk;

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
   Calib calib;

   bool isPreintegrated;
private:
    Preintegration(){Initialize(Bias(0,0,0,0,0,0));};

    Preintegration(const Bias &b_,Calib ImuCalib_,const Imu::Ptr imu)
{
    calib=ImuCalib_;
    Nga =ImuCalib_.Cov;
    NgaWalk = ImuCalib_.CovWalk;
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