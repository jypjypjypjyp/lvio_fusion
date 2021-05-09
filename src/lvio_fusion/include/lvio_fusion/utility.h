#ifndef lvio_fusion_UTILITY_H
#define lvio_fusion_UTILITY_H

// utilities used in lvio_fusion
#include "lvio_fusion/common.h"
#include <algorithm>
#include <cmath>

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

// *******************************Visual*******************************
/**
 * linear triangulation with SVD
 * @param pose0     pose,
 * @param pose1     pose,
 * @param p0        point in normalized plane
 * @param p1        point in normalized plane
 * @param p_3d      triangulated point in the world
 */
void triangulate(const SE3d &pose0, const SE3d &pose1, const Vector3d &p0, const Vector3d &p1, Vector3d &p_3d);

double cv_distance(cv::Point2f &pt1, cv::Point2f &pt2);

/**
 * double calculate optical flow
 * @param prevImg     prev image,
 * @param nextImg     next image,
 * @param prevPts     point in prev image
 * @param nextPts     point in next image
 * @param status      status
 */
int optical_flow(cv::Mat &prevImg, cv::Mat &nextImg,
                        std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                        std::vector<uchar> &status);

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

// *******************************Lidar********************************
void se32rpyxyz(const SE3d relatice_i_j, double *rpyxyz);

SE3d rpyxyz2se3(const double *rpyxyz);

template <typename PointT>
inline void filter_points_by_distance(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, float min_range, float max_range)
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

// *******************************Imu**********************************
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

template <typename Derived>
inline Matrix<typename Derived::Scalar, 3, 3> skew_symmetric(const MatrixBase<Derived> &q)
{
    Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
inline Matrix<typename Derived::Scalar, 4, 4> q_left(const QuaternionBase<Derived> &q)
{
    Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();
    ans.template block<3, 1>(1, 0) = q.vec(), ans.template block<3, 3>(1, 1) = q.w() * Matrix<typename Derived::Scalar, 3, 3>::Identity() + skew_symmetric(q.vec());
    return ans;
}

template <typename Derived>
inline Matrix<typename Derived::Scalar, 4, 4> q_right(const QuaternionBase<Derived> &p)
{
    Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
    ans.template block<3, 1>(1, 0) = p.vec(), ans.template block<3, 3>(1, 1) = p.w() * Matrix<typename Derived::Scalar, 3, 3>::Identity() - skew_symmetric(p.vec());
    return ans;
}

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
Vector3d R2ypr(const Matrix3d &R);

/**
 * g2R
 * @param g
 * @return R
 */
Matrix3d g2R(const Vector3d &g);

// *******************************Navsat*******************************
SE3d get_pose_from_two_points(const Vector3d &a, const Vector3d &b);

// *******************************Geometry*****************************

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

double vectors_degree_angle(Vector3d v1, Vector3d v2);

// SE3d slerp, the bigger s(0,1) is, the closer the result is to b
SE3d se3_slerp(const SE3d &a, const SE3d &b, double s);

Matrix3d exp_so3(const Vector3d &so3);

Matrix3d normalize_R(const Matrix3d &R_);

Matrix3d get_R_from_vector(Vector3d vec);

} // namespace lvio_fusion

#endif // lvio_fusion_UTILITY_H
