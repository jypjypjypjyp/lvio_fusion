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

inline double cv_distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * double calculate optical flow
 * @param prevImg     prev image,
 * @param nextImg     next image,
 * @param prevPts     point in prev image
 * @param nextPts     point in next image
 * @param status      status
 */
inline int optical_flow(cv::Mat &prevImg, cv::Mat &nextImg,
                        std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                        std::vector<uchar> &status)
{
    if (prevPts.empty())
        return 0;

    cv::Mat err;
    cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = prevPts;
    cv::calcOpticalFlowPyrLK(
        nextImg, prevImg, nextPts, reverse_pts, reverse_status, err, cv::Size(3, 3), 1,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_success_pts = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] && reverse_status[i] &&
            cv_distance(prevPts[i], reverse_pts[i]) <= 0.5 &&
            nextPts[i].x >= 0 && nextPts[i].x < prevImg.cols &&
            nextPts[i].y >= 0 && nextPts[i].y < prevImg.rows)
        {
            status[i] = 1;
            num_success_pts++;
        }
        else
            status[i] = 0;
    }
    return num_success_pts;
}

inline Vector2d cv2eigen(const cv::Point2f &p) { return Vector2d(p.x, p.y); }
inline Vector3d cv2eigen(const cv::Point3f &p) { return Vector3d(p.x, p.y, p.z); }
inline cv::Point2f eigen2cv(const Vector2d &p) { return cv::Point2f(p.x(), p.y()); }
inline cv::Point3f eigen2cv(const Vector3d &p) { return cv::Point3f(p.x(), p.y(), p.z()); }

// convert opencv points
inline void convert_points(const std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &ps)
{
    ps.resize(kps.size());
    for (int i = 0; i < kps.size(); i++)
    {
        ps[i] = kps[i].pt;
    }
}

// convert opencv points
inline void convert_points(const std::vector<cv::Point2f> &ps, std::vector<cv::KeyPoint> &kps)
{
    kps.resize(ps.size());
    for (int i = 0; i < ps.size(); i++)
    {
        kps[i] = cv::KeyPoint(ps[i], 1);
    }
}

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

    int j = 0;
    for (int i = 0; i < cloud_in.points.size(); ++i)
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

// SE3d slerp, the bigger s(0,1), the closer the result is to b
inline SE3d se3_slerp(const SE3d &a, const SE3d &b, double s)
{
    Quaterniond q = a.unit_quaternion().slerp(s, b.unit_quaternion());
    Vector3d t = s * b.translation() + (1 - s) * a.translation();
    return SE3d(q, t);
}

inline Matrix3d exp_so3(const Vector3d &so3)
{
    Matrix3d I = Matrix3d::Identity();
    const double d = so3.norm();
    const double d2 = d * d;
    Matrix3d W;
    W << 0, -so3.z(), so3.y(),
        so3.z(), 0, -so3.x(),
        -so3.y(), so3.x(), 0;
    if (d < 1e-4)
        return (I + W + 0.5f * W * W);
    else
        return (I + W * sin(d) / d + W * W * (1.0f - cos(d)) / d2);
}

inline Matrix3d normalize_R(const Matrix3d &R_)
{
    cv::Mat_<double> U, w, Vt;
    cv::Mat_<double> R = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2), R_(1, 0), R_(1, 1), R_(1, 2), R_(2, 0), R_(2, 1), R_(2, 2));
    cv::SVDecomp(R, w, U, Vt, cv::SVD::FULL_UV);
    Matrix3d uvt;
    cv::cv2eigen(U * Vt, uvt);
    return uvt;
}

} // namespace lvio_fusion

#endif // lvio_fusion_UTILITY_H
