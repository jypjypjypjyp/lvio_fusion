#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>

namespace lvio_fusion
{
void  Initializer::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    float minTime=1.0;  // 初始化需要的最小时间间隔
    int nMinKF=10;     // 初始化需要的最少关键帧数
    if(map_->GetAllKeyFramesSize()<nMinKF)
        return;
   // Step 1:按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > lpKF;
    Frames keyframes=map_->GetAllKeyFrames();
    Frames::reverse_iterator   iter;
     for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++){
          lpKF.push_front(iter->second);
     }
    std::vector< Frame::Ptr > vpKF(lpKF.begin(),lpKF.end());
    if(vpKF.size()<nMinKF)
        return;
    
       // imu计算初始时间
    mFirstTs=vpKF.front()->time;
    if(map_->current_frame->time-mFirstTs<minTime)
        return;

    bInitializing = true;   // 暂时未使用

    const int N = vpKF.size();  // 待处理的关键帧数目
    Bias b(0,0,0,0,0,0);

    // Step 2:估计KF速度和重力方向
    if (!initialized)
    {
        cv::Mat cvRwg;
        cv::Mat dirG = cv::Mat::zeros(3,1,CV_32F);  // 重力方向
        for(std::vector<Frame::Ptr>::iterator itKF = vpKF.begin()+1; itKF!=vpKF.end(); itKF++) 
        {
            if (!(*itKF)->preintegration)
                continue;
            // 预积分中delta_V 用来表示:Rwb_i.transpose()*(V2 - V1 - g*dt),故此处获得 -(V_i - V_0 - (i-0)*(mRwg*gI)*dt)
            // 应该使用将速度偏差在此处忽略或当做噪声，因为后面会优化mRwg
            dirG -= (*(itKF-1))->GetImuRotation()*(*itKF)->Preintegrated->GetUpdatedDeltaVelocity();
            cv::Mat _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        // Step 2.1:计算重力方向(与z轴偏差)，用轴角方式表示偏差
        dirG = dirG/cv::norm(dirG);
        cv::Mat gI = (cv::Mat_<float>(3,1) << 0.0f, 0.0f, -1.0f); //沿-z的归一化的重力数值
        // dirG和gI的模长都是1,故cross为sin，dot为cos
        
        // 计算旋转轴
        cv::Mat v = gI.cross(dirG);
        const float nv = cv::norm(v);
        // 计算旋转角
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        cv::Mat vzg = v*ang/nv;
        cvRwg = IMU::ExpSO3(vzg);
        mRwg = Converter::toMatrix3d(cvRwg);
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = Converter::toVector3d(mpCurrentKeyFrame->GetGyroBias());
        mba = Converter::toVector3d(mpCurrentKeyFrame->GetAccBias());
    }
 
    mScale=1.0;

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;
    
    // Step 3:进行惯性优化
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // 使用camera初始地图frame的pose与预积分的差值优化
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, false, Eigen::MatrixXd::Zero(9,9), false, false, priorG, priorA);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    /*cout << "scale after inertial-only optimization: " << mScale << endl;
    cout << "bg after inertial-only optimization: " << mbg << endl;
    cout << "ba after inertial-only optimization: " << mba << endl;*/

    // 如果求解的scale过小，跳过，下次在优化
    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }




    // Before this line we are not changing the map
    // 上面的程序没有改变地图，下面会对地图进行修改

    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // Step 4:在非双目，且scale有变化时，更新地图
    if ((fabs(mScale-1.f)>0.00001)||!mbMonocular)
    {
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(),mScale,true);
        mpTracker->UpdateFrameIMU(mScale,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // Check if initialization OK
    // Step 4:更新关键帧中imu状态
    if (!mpAtlas->isImuInitialized())
        for(int i=0;i<N;i++)
        {
            KeyFrame* pKF2 = vpKF[i];
            pKF2->bImu = true;
        }

    /*cout << "Before GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;*/

    // Step 5: 进行完全惯性优化(包括MapPoints)
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    // Step 6: 设置当前map imu 已经初始化 
    // If initialization is OK
    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        cout << "IMU in Map " << mpAtlas->GetCurrentMap()->GetId() << " is initialized" << endl;
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;  // 设置imu初始化时间
        mpCurrentKeyFrame->bImu = true;
    }

    //更新记录初始化状态的变量
    mbNewInit=true;
    mnKFs=vpKF.size();
    mIdxInit++;

    // Step 7: 清除KF待处理列表中剩余的KF
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;


    /*cout << "After GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;
    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
    double t_update = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
    double t_viba = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
    cout << t_inertial_only << ", " << t_update << ", " << t_viba << endl;*/

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;

}







/*bool Initializer::Initialize(Frames kfs)
{
    // be perpare for initialization
    std::vector<Initializer::Frame> frames;
    for (auto kf_pair : kfs)
    {
        if (!kf_pair.second->preintegration)
            return false;
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
    LOG(INFO) << "IMU Initialization failed.";
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

void Initializer::SolveGyroscopeBias(std::vector<Initializer::Frame> &frames)
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
*/
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