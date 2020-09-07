#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>

namespace lvio_fusion
{
bool Initializer::Initialize(Frames kfs)
{
    // be perpare for initialization
    std::vector<Initializer::Frame> frames;
    for (auto kf_pair : kfs)
    {
        Initializer::Frame frame;
        frame.preintegration = kf_pair.second->preintegration;
        frame.R = kf_pair.second->pose.inverse().rotationMatrix();
        frame.T = kf_pair.second->pose.inverse().translation();
        frame.Ba = kf_pair.second->preintegration->linearized_ba;
        frame.Bg = kf_pair.second->preintegration->linearized_bg;
        frames.push_back(frame);
    }

    SolveGyroscopeBias(frames);
    for (auto frame : frames)
    {
        frame.preintegration->Repropagate(Vector3d::Zero(), frame.Bg);
    }
    initialized = true;
    return true;
    // //check imu observibility
    // Frames::iterator frame_it;
    // Vector3d sum_g;
    // for (frame_it = frames_.begin(); next(frame_it) != frames_.end(); frame_it++)
    // {
    //     double dt = frame_it->second->preintegration->sum_dt;
    //     Vector3d tmp_g = frame_it->second->preintegration->delta_v / dt;
    //     sum_g += tmp_g;
    // }
    // Vector3d aver_g;
    // aver_g = sum_g * 1.0 / ((int)frames_.size() - 1);
    // double var = 0;
    // for (frame_it = frames_.begin(); next(frame_it) != frames_.end(); frame_it++)
    // {
    //     double dt = frame_it->second->preintegration->sum_dt;
    //     Vector3d tmp_g = frame_it->second->preintegration->delta_v / dt;
    //     var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    // }
    // var = sqrt(var / ((int)frames_.size() - 1));
    // if (var < 0.25)
    // {
    //     LOG(INFO) << "IMU excitation not enouth!";
    //     return false;
    // }

    // // visual initial align
    // if (VisualInitialAlign())
    //     return true;
    // else
    // {
    //     LOG(INFO) << "misalign visual structure with IMU";
    //     return false;
    // }
}

// bool Initializer::VisualInitialAlign()
// {
//     VectorXd x;
//     //solve scale
//     bool result = VisualIMUAlignment(x);
//     if (!result)
//     {
//         //ROS_DEBUG("solve g_ failed!");
//         return false;
//     }

//     double s = (x.tail<1>())(0);
//     Frames::iterator frame_i;
//     for (frame_i = frames_.begin(); frame_i != frames_.end(); frame_i++)
//     {
//         frame_i->second->preintegration->Repropagate(Vector3d::Zero(), frame_i->second->preintegration->Bg);
//     }

//     Matrix3d R0 = g2R(g_);
//     double yaw = R2ypr(R0 * frames_.begin()->second->pose.rotationMatrix()).x();
//     R0 = ypr2R(Vector3d{-yaw, 0, 0}) * R0;
//     g_ = R0 * g_;

//     Matrix3d rot_diff = R0;

//     return true;
// }

void Initializer::SolveGyroscopeBias(std::vector<Initializer::Frame> frames)
{
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();
    Vector3d delta_bg;

    for (int i = 0, j = 1; j < frames.size() - 1; i++, j++)
    {
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Quaterniond q_ij(frames[i].R.transpose() * frames[j].R);
        tmp_A = frames[i].preintegration->jacobian.template block<3, 3>(imu::O_R, imu::O_BG);
        tmp_b = 2 * (frames[i].preintegration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);

    for (int i = 0; i < frames.size() - 1; i++)
    {
        frames[i].Bg += delta_bg;
        frames[i].preintegration->Repropagate(Vector3d::Zero(), frames[0].Bg);
    }
}

// inline MatrixXd TangentBasis(Vector3d &g0)
// {
//     Vector3d b, c;
//     Vector3d a = g0.normalized();
//     Vector3d tmp(0, 0, 1);
//     if (a == tmp)
//         tmp << 1, 0, 0;
//     b = (tmp - a * (a.transpose() * tmp)).normalized();
//     c = a.cross(b);
//     MatrixXd bc(3, 2);
//     bc.block<3, 1>(0, 0) = b;
//     bc.block<3, 1>(0, 1) = c;
//     return bc;
// }

// void Initializer::RefineGravity(VectorXd &x)
// {
//     Vector3d g0 = g_.normalized() * imu::g.norm();
//     Vector3d lx, ly;
//     int all_frame_count = frames_.size();
//     int n_state = all_frame_count * 3 + 2 + 1;

//     MatrixXd A{n_state, n_state};
//     A.setZero();
//     VectorXd b{n_state};
//     b.setZero();

//     Frames::iterator frame_i;
//     Frames::iterator frame_j;
//     for (int k = 0; k < 4; k++)
//     {
//         MatrixXd lxly(3, 2);
//         lxly = TangentBasis(g0);
//         int i = 0;
//         for (frame_i = frames_.begin(); next(frame_i) != frames_.end(); frame_i++, i++)
//         {
//             frame_j = next(frame_i);

//             MatrixXd tmp_A(6, 9);
//             tmp_A.setZero();
//             VectorXd tmp_b(6);
//             tmp_b.setZero();

//             double dt = frame_i->second->preintegration->sum_dt;

//             tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
//             tmp_A.block<3, 2>(0, 6) = frame_i->second->pose.rotationMatrix() * dt * dt / 2 * Matrix3d::Identity() * lxly;
//             tmp_A.block<3, 1>(0, 8) = frame_i->second->pose.rotationMatrix() * (frame_j->second->pose.inverse().translation() - frame_i->second->pose.inverse().translation()) / 100.0;
//             tmp_b.block<3, 1>(0, 0) = frame_i->second->preintegration->delta_p - frame_i->second->pose.rotationMatrix() * dt * dt / 2 * g0;

//             tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
//             tmp_A.block<3, 3>(3, 3) = frame_i->second->pose.rotationMatrix() * frame_j->second->pose.rotationMatrix().transpose();
//             tmp_A.block<3, 2>(3, 6) = frame_i->second->pose.rotationMatrix() * dt * Matrix3d::Identity() * lxly;
//             tmp_b.block<3, 1>(3, 0) = frame_i->second->preintegration->delta_v - frame_i->second->pose.rotationMatrix() * dt * Matrix3d::Identity() * g0;

//             // NOTE: remove useless cov_inv
//             MatrixXd r_A = tmp_A.transpose() * tmp_A;
//             VectorXd r_b = tmp_A.transpose() * tmp_b;

//             A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
//             b.segment<6>(i * 3) += r_b.head<6>();

//             A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
//             b.tail<3>() += r_b.tail<3>();

//             A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
//             A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
//         }
//         A = A * 1000.0;
//         b = b * 1000.0;
//         x = A.ldlt().solve(b);
//         VectorXd dg = x.segment<2>(n_state - 3);
//         g0 = (g0 + lxly * dg).normalized() * imu::g.norm();
//     }
//     g_ = g0;
// }

// bool Initializer::LinearAlignment(VectorXd &x)
// {
//     int all_frame_count = frames_.size();
//     int n_state = all_frame_count * 3 + 3 + 1;

//     MatrixXd A{n_state, n_state};
//     A.setZero();
//     VectorXd b{n_state};
//     b.setZero();

//     Frames::iterator frame_i;
//     Frames::iterator frame_j;
//     int i = 0;
//     for (frame_i = frames_.begin(); next(frame_i) != frames_.end(); frame_i++, i++)
//     {
//         frame_j = next(frame_i);

//         MatrixXd tmp_A(6, 10);
//         tmp_A.setZero();
//         VectorXd tmp_b(6);
//         tmp_b.setZero();

//         double dt = frame_i->second->preintegration->sum_dt;

//         tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
//         tmp_A.block<3, 3>(0, 6) = frame_i->second->pose.rotationMatrix() * dt * dt / 2 * Matrix3d::Identity();
//         tmp_A.block<3, 1>(0, 9) = frame_i->second->pose.rotationMatrix() * (frame_j->second->pose.inverse().translation() - frame_i->second->pose.inverse().translation()) / 100.0;
//         tmp_b.block<3, 1>(0, 0) = frame_i->second->preintegration->delta_p;
//         tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
//         tmp_A.block<3, 3>(3, 3) = frame_i->second->pose.rotationMatrix() * frame_j->second->pose.rotationMatrix().transpose();
//         tmp_A.block<3, 3>(3, 6) = frame_i->second->pose.rotationMatrix() * dt * Matrix3d::Identity();
//         tmp_b.block<3, 1>(3, 0) = frame_i->second->preintegration->delta_v;

//         // NOTE: remove useless con_inv
//         MatrixXd r_A = tmp_A.transpose() * tmp_A;
//         VectorXd r_b = tmp_A.transpose() * tmp_b;

//         A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
//         b.segment<6>(i * 3) += r_b.head<6>();

//         A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
//         b.tail<4>() += r_b.tail<4>();

//         A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
//         A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
//     }
//     A = A * 1000.0;
//     b = b * 1000.0;
//     x = A.ldlt().solve(b);
//     double s = x(n_state - 1) / 100.0; //scale
//     g_ = x.segment<3>(n_state - 4);    // g_
//     if (fabs(g_.norm() - imu::g.norm()) > 1.0 || s < 0)
//     {
//         return false;
//     }

//     RefineGravity(x);
//     s = (x.tail<1>())(0) / 100.0;
//     (x.tail<1>())(0) = s;
//     if (s < 0.0)
//         return false;
//     else
//         return true;
// }

// bool Initializer::VisualIMUAlignment(VectorXd &x)
// {
//     SolveGyroscopeBias();

//     if (LinearAlignment(x))
//         return true;
//     else
//         return false;
// }
} // namespace lvio_fusion