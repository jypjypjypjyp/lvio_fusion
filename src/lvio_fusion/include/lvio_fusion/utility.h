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
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                          const std::vector<Vector3d> points, Vector3d &pt_world)
{
    MatrixXd A(2 * poses.size(), 4);
    VectorXd b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i)
    {
        Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (pt_world[2] < 0)
    {
        return false;
    }
    return true;
}

inline Vector2d to_vector2d(const cv::Point2f p) { return Vector2d(p.x, p.y); }

/**
 * line fitting
 * @param P    points set
 * @param A    A
 * @param B    B
 */
inline void line_fitting(MatrixX3d P, Vector3d &A, Vector3d &B)
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
inline Vector3d closest_point_on_a_line(Vector3d A, Vector3d B, Vector3d P)
{
    Vector3d AB = B-A, AP = P-A;
    double k = AB.dot(AP) / AB.norm();
    return A + k * AB;
};

} // namespace lvio_fusion

#endif // lvio_fusion_UTILITY_H
