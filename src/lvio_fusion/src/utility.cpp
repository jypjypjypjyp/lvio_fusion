#include "lvio_fusion/utility.h"
#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

void triangulate(const SE3d &pose0, const SE3d &pose1, const Vector3d &p0, const Vector3d &p1, Vector3d &p_3d)
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

double cv_distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void se32rpyxyz(const SE3d relatice_i_j, double *rpyxyz)
{
    ceres::EigenQuaternionToRPY(relatice_i_j.data(), rpyxyz);
    rpyxyz[3] = relatice_i_j.data()[4];
    rpyxyz[4] = relatice_i_j.data()[5];
    rpyxyz[5] = relatice_i_j.data()[6];
}

SE3d rpyxyz2se3(const double *rpyxyz)
{
    double e_q[4];
    ceres::RPYToEigenQuaternion(rpyxyz, e_q);
    return SE3d(Quaterniond(e_q), Vector3d(rpyxyz[3], rpyxyz[4], rpyxyz[5]));
}

SE3d get_pose_from_two_points(const Vector3d &a, const Vector3d &b)
{
    Vector3d d = b - a;
    double rpyxyz[6];
    rpyxyz[0] = atan2(d.y(), d.x());
    rpyxyz[1] = -atan2(d.z(), Vector2d(d.x(), d.y()).norm());
    rpyxyz[2] = 0;
    rpyxyz[3] = b.x();
    rpyxyz[4] = b.y();
    rpyxyz[5] = b.z();
    return rpyxyz2se3(rpyxyz);
}

void optical_flow(cv::Mat &prevImg, cv::Mat &nextImg,
                  std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                  std::vector<uchar> &status)
{
    if (prevPts.empty())
        return;

    cv::Mat err;
    cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = prevPts;
    cv::calcOpticalFlowPyrLK(
        nextImg, prevImg, nextPts, reverse_pts, reverse_status, err, cv::Size(3, 3), 3,
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
}

Vector3d R2ypr(const Matrix3d &R)
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

Matrix3d g2R(const Vector3d &g)
{
    Matrix3d R0;
    Vector3d ng1 = g.normalized();
    Vector3d ng2{0, 0, 1.0};
    R0 = Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Vector3d{-yaw, 0, 0}) * R0;
    return R0;
}

double vectors_degree_angle(Vector3d v1, Vector3d v2)
{
    double radian_angle = atan2(v1.cross(v2).norm(), v1.transpose() * v2);
    return radian_angle * 180 / M_PI;
}

SE3d se3_slerp(const SE3d &a, const SE3d &b, double s)
{
    Quaterniond q = a.unit_quaternion().slerp(s, b.unit_quaternion());
    Vector3d t = s * b.translation() + (1 - s) * a.translation();
    return SE3d(q, t);
}

Matrix3d exp_so3(const Vector3d &so3)
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

Matrix3d normalize_R(const Matrix3d &R_)
{
    cv::Mat_<double> U, w, Vt;
    cv::Mat_<double> R = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2), R_(1, 0), R_(1, 1), R_(1, 2), R_(2, 0), R_(2, 1), R_(2, 2));
    cv::SVDecomp(R, w, U, Vt, cv::SVD::FULL_UV);
    Matrix3d uvt;
    cv::cv2eigen(U * Vt, uvt);
    return uvt;
}

Matrix3d get_R_from_vector(Vector3d vec)
{
    vec.normalize();
    Vector3d v = Vector3d::UnitZ().cross(vec);
    double cosg = Vector3d::UnitZ().dot(vec);
    double ang = acos(cosg);
    Vector3d vzg = v * ang / v.norm();
    return exp_so3(vzg);
}
} // namespace lvio_fusion