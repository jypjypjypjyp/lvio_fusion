#ifndef lvio_fusion_SE3d_PARAMETERIZATION_H
#define lvio_fusion_SE3d_PARAMETERIZATION_H

#include <ceres/ceres.h>
#include "lvio_fusion/common.h"

namespace lvio_fusion
{
class SE3Parameterization : public ceres::LocalParameterization
{
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    // SE3d plus operation for Ceres
    //
    // exp(delta)*T
    //
    virtual bool Plus(double const *T_raw, double const *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<SE3d const> const T(T_raw);
        Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
        Eigen::Map<Vector3d const> delta_phi(delta_raw);
        Eigen::Map<Vector3d const> delta_rho(delta_raw+3);
        Matrix<double,6,1> delta;
        delta.block<3,1>(0,0) = delta_rho;//translation
        delta.block<3,1>(3,0) = delta_phi;//rotation
        T_plus_delta = SE3d::exp(delta) * T;
        return true;
    }

    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<SE3d const> T(T_raw);
        Eigen::Map<Matrix<double, 7, 6,RowMajor>> jacobian(jacobian_raw);
        jacobian.setZero();
        jacobian.block<3,3>(0,0) = Matrix3d::Identity();
        jacobian.block<3,3>(4,3) = Matrix3d::Identity();
        return true;
    }

    virtual int GlobalSize() const { return SE3d::num_parameters; }
    virtual int LocalSize() const {return SE3d::DoF; }
};

} // namespace lvio_fusion

#endif // lvio_fusion_SE3d_PARAMETERIZATION_H