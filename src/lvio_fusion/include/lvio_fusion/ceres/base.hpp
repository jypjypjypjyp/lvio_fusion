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

} // namespace ceres

#endif // lvio_fusion_BASE_H
