#ifndef lvio_fusion_BASE_H
#define lvio_fusion_BASE_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace ceres
{
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
    result[0] += se3[4];
    result[1] += se3[5];
    result[2] += se3[6];
}

template <typename T>
inline void EigenQuaternionInverse(const T e_q[4], T e_q_inverse[4])
{
    e_q_inverse[0] = -e_q[0];
    e_q_inverse[1] = -e_q[1];
    e_q_inverse[2] = -e_q[2];
    e_q_inverse[3] = e_q[3];
};

template <typename T>
inline void SE3Inverse(const T se3[7], T se3_inverse[7])
{
    EigenQuaternionInverse(se3, se3_inverse);
    T translation_inverse[3] = {-se3[4], -se3[5], -se3[6]};
    EigenQuaternionRotatePoint(se3_inverse, translation_inverse, se3_inverse + 4);
}

template <typename T>
inline void Cast(double *raw, int size, T *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = T(raw[i]);
    }
}

} // namespace ceres

#endif // lvio_fusion_BASE_H
