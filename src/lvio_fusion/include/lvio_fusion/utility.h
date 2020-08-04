#ifndef lvio_fusion_UTILITY_H
#define lvio_fusion_UTILITY_H

// utilities used in lvio_fusion
#include "lvio_fusion/common.h"
#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

/**
 * linear triangulation with SVD
 * @param pose0     pose,
 * @param pose1     pose,
 * @param p0        point in normalized plane
 * @param p1        point in normalized plane
 * @param p_3d      triangulated point in the world
 */
inline void triangulate(const SE3d &pose0, const SE3d &pose1, const Vector3d &p0, const Vector3d &p1, Vector3d &p_3d)
{
    Matrix4d A = Matrix4d::Zero();
    Matrix<double, 3, 4> P0 = pose0.matrix3x4();
    Matrix<double, 3, 4> P1 = pose1.matrix3x4();
    A.row(0) = p0[0] * P0.row(2) - P0.row(0);
    A.row(1) = p0[1] * P0.row(2) - P0.row(1);
    A.row(2) = p1[0] * P1.row(2) - P1.row(0);
    A.row(3) = p1[1] * P1.row(2) - P1.row(1);
    Vector4d p_norm = A.jacobiSvd(ComputeFullV).matrixV().rightCols<1>();
    p_3d = (p_norm / p_norm(3)).head<3>();
}

inline Vector2d cv2eigen(const cv::Point2f &p) { return Vector2d(p.x, p.y); }

inline cv::Point2f eigen2cv(const Vector2d &p) { return cv::Point2f(p.x(), p.y()); }

/**
 * line fitting
 * @param P    points set
 * @param A    A
 * @param B    B
 */
inline void line_fitting(const MatrixX3d &P, Vector3d &A, Vector3d &B)
{
    A = P.colwise().mean();
    MatrixXd P0 = P.rowwise() - A.transpose();
    auto cov = (P0.adjoint() * P0) / double(P.rows() - 1);
    auto svd = cov.bdcSvd(Eigen::ComputeThinV);
    auto V = svd.matrixV();
    Vector3d v(V.block<3, 1>(0, 0));
    B = A + v * 1;
}

/**
 * closest point on a line
 * @param A    A
 * @param B    B
 * @param P    P
 * @return closest point
 */
inline Vector3d closest_point_on_a_line(const Vector3d &A, const Vector3d &B, const Vector3d &P)
{
    Vector3d AB = B - A, AP = P - A;
    double k = AB.dot(AP) / AB.norm();
    return A + k * AB;
};

} // namespace lvio_fusion

#endif // lvio_fusion_UTILITY_H
