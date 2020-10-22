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

    void Appendimu(imuPoint imuMeasure)
    {
        imuData_buf.push_back(imuMeasure);
    }

    void PreintegrateIMU(double last_frame_time,double current_frame_time);
   void IntegrateNewMeasurement(const Vector3d &acceleration, const Vector3d  &angVel, const float &dt);
    void Initialize(const Bias &b_);
    cv::Mat GetUpdatedDeltaVelocity();
    void Preintegration::SetNewBias(const Bias &bu_);

  /*  double dt;
    Vector3d acc0, gyr0;
    Vector3d acc1, gyr1;
    Vector3d linearized_acc, linearized_gyr;
    Vector3d linearized_ba, linearized_bg;
    Matrix<double, 15, 15> jacobian, covariance;
    Matrix<double, 15, 15> step_jacobian;
    Matrix<double, 15, 18> step_V;
    Matrix<double, 18, 18> noise;
    double sum_dt;
    Vector3d delta_p;
    Quaterniond delta_q;
    Vector3d delta_v;
    Vector3d v0;
*/
   std::vector<imuPoint> imuData_buf;

    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf;
    std::vector<Vector3d> gyr_buf;



    float dT;
    cv::Mat C;   //cov
    cv::Mat Info;
    cv::Mat Nga, NgaWalk;

    // Values for the original bias (when integration was computed)
    Bias b;
    cv::Mat dR, dV, dP;
    cv::Mat JRg, JVg, JVa, JPg, JPa; 
    cv::Mat avgA;
    cv::Mat avgW;
    
   // Updated bias
    Bias bu;
    // Dif between original and updated bias
    // This is used to compute the updated values of the preintegration
    cv::Mat db;
   Calib calib;
private:
    Preintegration() = default;

    Preintegration(const Bias &b_,Calib ImuCalib_,const Imu::Ptr imu)//TODO:Imu::Ptr imu暂未使用 如果全局参数的方法不可行 可以将其作为Calib使用
{
    calib=ImuCalib_;
    Nga =ImuCalib_.Cov.clone();
    NgaWalk = ImuCalib_.CovWalk.clone();
    Initialize(b_);
}

    struct integrable
    {
        integrable(const Vector3d &a_, const Vector3d &w_ , const float &t_):a(a_),w(w_),t(t_){}
        Vector3d a;
        Vector3d w;
        float t;
    };
    std::vector<integrable> mvMeasurements;


};

typedef std::map<double, Preintegration::Ptr> PreIntegrations;
} // namespace imu

} // namespace lvio_fusion

#endif // lvio_fusion_PREINTEGRATION_H