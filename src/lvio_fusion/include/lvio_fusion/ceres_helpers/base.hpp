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
inline void EigenQuaternionInverse(const T e_q[4], T e_q_inverse[4])
{
    e_q_inverse[0] = -e_q[0];
    e_q_inverse[1] = -e_q[1];
    e_q_inverse[2] = -e_q[2];
    e_q_inverse[3] = e_q[3];
};

} // namespace ceres

#endif // lvio_fusion_BASE_H
