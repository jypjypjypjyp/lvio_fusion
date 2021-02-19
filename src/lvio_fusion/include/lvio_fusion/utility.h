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
inline Vector3d cv2eigen(const cv::Point3f &p) { return Vector3d(p.x, p.y, p.z); }
inline cv::Point2f eigen2cv(const Vector2d &p) { return cv::Point2f(p.x(), p.y()); }
inline cv::Point3f eigen2cv(const Vector3d &p) { return cv::Point3f(p.x(), p.y(), p.z()); }

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
    auto svd = cov.bdcSvd(ComputeThinV);
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

/**
 * closest point on a plane
 * @param norm norm
 * @param A    A
 * @param P    P
 * @return closest point
 */
inline Vector3d closest_point_on_a_panel(const Vector3d &norm, const Vector3d &A, const Vector3d &P)
{
    auto t = (A - P).dot(norm) / norm.squaredNorm();
    return P + t * norm;
};

/**
 * remove close points
 * @param cloud_in      input
 * @param cloud_out     output
 * @param thres         threshold
 */
template <typename PointT>
void filter_points_by_distance(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, float min_range, float max_range)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        float d = cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z;
        if (d > min_range * min_range && d < max_range * max_range)
        {
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

/**
 * delta Q
 * @param theta
 * @return delta Q
 */
template <typename Derived>
inline Quaternion<typename Derived::Scalar> q_delta(const MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Quaternion<Scalar_t> dq;
    Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

/**
 * skewSymmetric
 * @param q
 * @return skewSymmetric
 */
template <typename Derived>
inline Matrix<typename Derived::Scalar, 3, 3> skew_symmetric(const MatrixBase<Derived> &q)
{
    Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

/**
 * Q left
 * @param q
 * @return Q left
 */
template <typename Derived>
inline Matrix<typename Derived::Scalar, 4, 4> q_left(const QuaternionBase<Derived> &q)
{
    Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();
    ans.template block<3, 1>(1, 0) = q.vec(), ans.template block<3, 3>(1, 1) = q.w() * Matrix<typename Derived::Scalar, 3, 3>::Identity() + skew_symmetric(q.vec());
    return ans;
}

/**
 * Q right
 * @param q
 * @return Q right
 */
template <typename Derived>
inline Matrix<typename Derived::Scalar, 4, 4> q_right(const QuaternionBase<Derived> &p)
{
    Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
    ans.template block<3, 1>(1, 0) = p.vec(), ans.template block<3, 3>(1, 1) = p.w() * Matrix<typename Derived::Scalar, 3, 3>::Identity() - skew_symmetric(p.vec());
    return ans;
}

/**
 * ypr2R
 * @param ypr
 * @return R
 */
template <typename Derived>
inline Matrix<typename Derived::Scalar, 3, 3> ypr2R(const MatrixBase<Derived> &ypr)
{
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

/**
 * R2ypr
 * @param R
 * @return ypr
 */
inline Vector3d R2ypr(const Matrix3d &R)
{
    Vector3d n = R.col(0);
    Vector3d o = R.col(1);
    Vector3d a = R.col(2);

    Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

/**
 * g2R
 * @param g
 * @return R
 */
inline Matrix3d g2R(const Vector3d &g)
{
    Matrix3d R0;
    Vector3d ng1 = g.normalized();
    Vector3d ng2{0, 0, 1.0};
    R0 = Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Vector3d{-yaw, 0, 0}) * R0;
    return R0;
}

/**
 * normalize Angle
 * @param degrees
 * @return normalize Angle
 */
template <typename T>
inline T normalize_angle(const T &angle_degrees)
{
    T two_pi(2.0 * 180);
    if (angle_degrees > 0)
        return angle_degrees -
               two_pi * std::floor((angle_degrees + T(180)) / two_pi);
    else
        return angle_degrees +
               two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
};

inline double vectors_degree_angle(Vector3d v1, Vector3d v2)
{
    double radian_angle = atan2(v1.cross(v2).norm(), v1.transpose() * v2);
    return radian_angle * 180 / M_PI;
}

inline double vectors_height(Vector3d v1, Vector3d v2)
{
    return v1.cross(v2).norm() / v1.norm();
}

inline double vectors_height(Vector3d A, Vector3d B, Vector3d C)
{
    auto v1 = B - A, v2 = C - A;
    return v1.cross(v2).norm() / v1.norm();
}

inline double panel_height(Vector3d A, Vector3d B, Vector3d C)
{
    A.z() = B.z() = C.z() = 0;
    return vectors_height(A, B, C);
}

inline double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// SE3d slerp, the bigger s(0,1), the closer the result is to b
inline SE3d se3_slerp(const SE3d &a, const SE3d &b, double s)
{   
    Quaterniond q = a.unit_quaternion().slerp(s, b.unit_quaternion());
    Vector3d t = s * b.translation() + (1 - s) * a.translation();
    return SE3d(q, t);
}
//IMU
inline Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

inline Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

inline Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}


inline Matrix3d ExpSO3(const double &x, const double &y, const double &z)
{
    Matrix3d I = Matrix3d::Identity();
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
   Matrix3d W;
   W << 0, -z, y,
            z, 0, -x,
            -y,  x, 0;
    if(d<1e-4)
        return (I + W + 0.5f*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2);
}

inline Matrix3d ExpSO3(const Vector3d &v)
{
    return ExpSO3(v(0),v(1),v(2));
}

inline Matrix3d NormalizeRotation(const Matrix3d &R_)
{ 
    cv::Mat_<double> U,w,Vt;
    cv::Mat_<double> R=(cv::Mat_<double>(3,3)<<R_(0,0),R_(0,1),R_(0,2),R_(1,0),R_(1,1),R_(1,2),R_(2,0),R_(2,1),R_(2,2));
    cv::SVDecomp(R,w,U,Vt,cv::SVD::FULL_UV);
    Matrix3d uvt;
    cv::cv2eigen(U*Vt,uvt);
    return uvt;
}

//IMUEND
} // namespace lvio_fusion

#endif // lvio_fusion_UTILITY_H
