#ifndef lvio_fusion_PREINTEGRATION_H
#define lvio_fusion_PREINTEGRATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.h"

namespace lvio_fusion
{

namespace imu
{

extern int O_T, O_R, O_V, O_BA, O_BG, O_PR, O_PT;
extern Vector3d g;

class Preintegration
{
public:
    typedef std::shared_ptr<Preintegration> Ptr;

    static Preintegration::Ptr Create(const Bias bias)
    {
        Preintegration::Ptr new_preintegration(new Preintegration(bias.linearized_ba, bias.linearized_bg));
        return new_preintegration;
    }
    
    void Append(double dt, const Vector3d &acc, const Vector3d &gyr, const Vector3d &acc0_, const Vector3d &gyr0_)
    {
        if (dt_buf.size() == 0)
        {
            acc0 = acc0_;
            gyr0 = gyr0_;
            linearized_acc = acc0_;
            linearized_gyr = gyr0_;
        }
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        Propagate(dt, acc, gyr);
    }
    
    void MidPointIntegration(double _dt,
                             const Vector3d &_acc_0, const Vector3d &_gyr_0,
                             const Vector3d &_acc_1, const Vector3d &_gyr_1,
                             const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                             const Vector3d &linearized_ba, const Vector3d &linearized_bg,
                             Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
                             Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian);

    void Propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1);
    void Repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg);

    Matrix<double, 15, 1> Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                   const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj);
    Matrix<double, 15, 1> Evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                   const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj, const Quaterniond &Rg);

    Vector3d GetUpdatedDeltaVelocity();
    void SetNewBias(const Bias &new_bias);
    Quaterniond GetUpdatedDeltaRotation();
    Vector3d GetUpdatedDeltaPosition();
    Quaterniond GetDeltaRotation(const Bias &b_);
    Vector3d GetDeltaVelocity(const Bias &b_);
    Vector3d GetDeltaPosition(const Bias &b_);
    Bias GetDeltaBias(const Bias &b_);

    double dt;
    double sum_dt;
    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf, gyr_buf;
    Vector3d acc0, gyr0;
    Vector3d acc1, gyr1;
    Vector3d linearized_acc, linearized_gyr;
    Vector3d linearized_ba, linearized_bg;

    Vector3d delta_p;
    Quaterniond delta_q;
    Vector3d delta_v;
    Bias bias;
    Matrix<double, 6, 1> delta_bias;
    bool isBad = false;

    Matrix<double, 15, 15> jacobian, covariance;
    Matrix<double, 15, 15> step_jacobian;
    Matrix<double, 15, 18> step_V;
    Matrix<double, 18, 18> noise;

private:
    Preintegration() = default;
    Preintegration(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg);
};

typedef std::map<double, Preintegration::Ptr> PreIntegrations;
} // namespace imu

} // namespace lvio_fusion

#endif // lvio_fusion_PREINTEGRATION_H