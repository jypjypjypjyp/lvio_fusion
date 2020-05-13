#ifndef lvio_fusion_SE3_PARAMETERIZATION_H
#define lvio_fusion_SE3_PARAMETERIZATION_H

#include <ceres/ceres.h>
#include "lvio_fusion/common.h"

namespace lvio_fusion
{
class SE3Parameterization : public ceres::LocalParameterization
{
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    // SE3 plus operation for Ceres
    //
    // exp(delta)*T
    //
    virtual bool Plus(double const *T_raw, double const *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<SE3 const> const T(T_raw);

        Eigen::Map<SE3> T_plus_delta(T_plus_delta_raw);

        Eigen::Map<Eigen::Vector3d const> delta_phi(delta_raw);
        Eigen::Map<Eigen::Vector3d const> delta_rho(delta_raw+3);
        Eigen::Matrix<double,6,1> delta;

        delta.block<3,1>(0,0) = delta_rho;//translation
        delta.block<3,1>(3,0) = delta_phi;//rotation

        T_plus_delta = SE3::exp(delta) * T;
        return true;
    }

    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<SE3 const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6,Eigen::RowMajor>> jacobian(jacobian_raw);
        jacobian.setZero();
        jacobian.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        jacobian.block<3,3>(4,3) = Eigen::Matrix3d::Identity();
        return true;
    }

    virtual int GlobalSize() const { return SE3::num_parameters; }
    virtual int LocalSize() const {return SE3::DoF; }
};

} // namespace lvio_fusion

#endif // lvio_fusion_SE3_PARAMETERIZATION_H