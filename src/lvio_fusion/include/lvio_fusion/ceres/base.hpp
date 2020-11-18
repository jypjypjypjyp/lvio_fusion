#ifndef lvio_fusion_BASE_H
#define lvio_fusion_BASE_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace ceres
{

template <typename T>
inline void Minus(const T A[3], const T B[3], T C[3])
{
    C[0] = A[0] - B[0];
    C[1] = A[1] - B[1];
    C[2] = A[2] - B[2];
}

template <typename T>
inline void Add(const T A[3], const T B[3], T C[3])
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + B[2];
}

template <typename T>
inline void EigenQuaternionRotatePoint(const T e_q[4], const T pt[3], T result[3])
{
    const T q[4] = {e_q[3], e_q[0], e_q[1], e_q[2]};
    QuaternionRotatePoint(q, pt, result);
}

template <typename T>
inline void SE3TransformPoint(const T se3[7], const T pt[3], T result[3])
{
    ceres::EigenQuaternionRotatePoint(se3, pt, result);
    Add(result, se3 + 4, result);
}

template <typename T>
inline void EigenQuaternionInverse(const T e_q[4], T e_q_inverse[4])
{
    e_q_inverse[0] = -e_q[0];
    e_q_inverse[1] = -e_q[1];
    e_q_inverse[2] = -e_q[2];
    e_q_inverse[3] = e_q[3];
}

template <typename T>
inline void SE3Inverse(const T se3[7], T se3_inverse[7])
{
    EigenQuaternionInverse(se3, se3_inverse);
    T translation_inverse[3] = {-se3[4], -se3[5], -se3[6]};
    EigenQuaternionRotatePoint(se3_inverse, translation_inverse, se3_inverse + 4);
}

template <typename T>
inline void EigenQuaternionProduct(const T e_z[4], const T e_w[4], T e_zw[4])
{
    const T z[4] = {e_z[3], e_z[0], e_z[1], e_z[2]};
    const T w[4] = {e_w[3], e_w[0], e_w[1], e_w[2]};
    T zw[4];
    ceres::QuaternionProduct(z, w, zw);
    e_zw[0] = zw[1];
    e_zw[1] = zw[2];
    e_zw[2] = zw[3];
    e_zw[3] = zw[0];
}

template <typename T>
inline void SE3Product(const T A[7], const T B[7], T C[7])
{
    ceres::EigenQuaternionProduct(A, B, C);
    T t[3];
    ceres::EigenQuaternionRotatePoint(A, B + 4, t);
    Add(A + 4, t, C + 4);
}

template <typename T>
inline void Norm(const T A[3], T *norm)
{
    *norm = sqrt(DotProduct(A, A));
}

template <typename T>
inline void Cast(const double *raw, int size, T *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = T(raw[i]);
    }
}

// rpy: Z->Y->X
template <typename T>
inline void QuaternionToRPY(const T *q, T *rpy)
{
    rpy[0] = atan2(T(2) * (q[1] * q[2] + q[0] * q[3]), 1 - T(2) * (q[2] * q[2] + q[3] * q[3]));
    rpy[1] = asin(T(2) * (q[0] * q[2] - q[1] * q[3]));
    rpy[2] = atan2(T(2) * (q[2] * q[3] + q[0] * q[1]), 1 - T(2) * (q[1] * q[1] + q[2] * q[2]));
};

template <typename T>
inline void EigenQuaternionToRPY(const T *e_q, T *rpy)
{
    const T q[4] = {e_q[3], e_q[0], e_q[1], e_q[2]};
    QuaternionToRPY(q, rpy);
};

template <typename T>
inline void RPYToQuaternion(const T *rpy, T *q)
{
    T z = rpy[0] / T(2), y = rpy[1] / T(2), x = rpy[2] / T(2);
    T c_z = cos(z), s_z = sin(z);
    T c_y = cos(y), s_y = sin(y);
    T c_x = cos(x), s_x = sin(x);
    q[0] = c_z * c_y * c_x + s_z * s_y * s_x;
    q[1] = c_z * c_y * s_x - s_z * s_y * c_x;
    q[2] = c_z * s_y * c_x + s_z * c_y * s_x;
    q[3] = s_z * c_y * c_x - c_z * s_y * s_x;
};

template <typename T>
inline void RPYToEigenQuaternion(const T *rpy, T *e_q)
{
    T q[4];
    RPYToQuaternion(rpy, q);
    e_q[0] = q[1];
    e_q[1] = q[2];
    e_q[2] = q[3];
    e_q[3] = q[0];
};

template <typename T>
inline void SE3ToRpyxyz(const T *relatice_i_j, T *rpyxyz)
{
    EigenQuaternionToRPY(relatice_i_j, rpyxyz);
    rpyxyz[3] = relatice_i_j[4];
    rpyxyz[4] = relatice_i_j[5];
    rpyxyz[5] = relatice_i_j[6];
}

template <typename T>
inline void RpyxyzToSE3(const T *rpyxyz, T *relatice_i_j)
{
    RPYToEigenQuaternion(rpyxyz, relatice_i_j);
    relatice_i_j[4] = rpyxyz[3];
    relatice_i_j[5] = rpyxyz[4];
    relatice_i_j[6] = rpyxyz[5];
}

} // namespace ceres

#endif // lvio_fusion_BASE_H
