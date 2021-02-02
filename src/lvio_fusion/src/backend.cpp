#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/manager.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{
     bool showIMUError(const double*  parameters0, const double*  parameters1, const double*  parameters2, const double*  parameters3, const double*  parameters4, const double*  parameters5, imu::Preintegration::Ptr mpInt, double time)  
    {
        Quaterniond Qi(parameters0[3], parameters0[0], parameters0[1], parameters0[2]);
        Vector3d Pi(parameters0[4], parameters0[5], parameters0[6]);
        Vector3d Vi(parameters1[0], parameters1[1], parameters1[2]);

        Vector3d gyroBias(parameters2[0], parameters2[1], parameters2[2]);
        Vector3d accBias(parameters3[0], parameters3[1],parameters3[2]);

        Quaterniond Qj(parameters4[3], parameters4[0], parameters4[1], parameters4[2]);
        Vector3d Pj(parameters4[4], parameters4[5], parameters4[6]);
        Vector3d Vj(parameters5[0], parameters5[1], parameters5[2]);
        double dt=(mpInt->dT);
        Vector3d g;
         g<< 0, 0, -G;
        // g=Rwg*g;
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = mpInt->GetDeltaRotation(b1);
        Vector3d dV = mpInt->GetDeltaVelocity(b1);
        Vector3d dP =mpInt->GetDeltaPosition(b1);
 
        const Vector3d er = LogSO3(dR.inverse()*Qi.toRotationMatrix().inverse()*Qj.toRotationMatrix());
        const Vector3d ev = Qi.toRotationMatrix().inverse()*((Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.toRotationMatrix().inverse()*((Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        Matrix<double, 9, 1> residual;
        residual<<er,ev,ep;
        //LOG(INFO)<<"InertialError residual "<<residual.transpose();
           Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
         Info = (Info+Info.transpose())/2;
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
         Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
         for(int i=0;i<9;i++)
             if(eigs[i]<1e-12)
                 eigs[i]=0;
         Matrix<double, 9,9> sqrt_info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
        //Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>( mpInt->C.block<9,9>(0,0).inverse()).matrixL().transpose();
      //  sqrt_info/=InfoScale;
        // LOG(INFO)<<"InertialError sqrt_info "<<sqrt_info;
        //assert(!isnan(residual[0])&&!isnan(residual[1])&&!isnan(residual[2])&&!isnan(residual[3])&&!isnan(residual[4])&&!isnan(residual[5])&&!isnan(residual[6])&&!isnan(residual[7])&&!isnan(residual[8]));
        residual = sqrt_info* residual;
        LOG(INFO)<<time<<"  IMUError:  r "<<residual.transpose()<<"  "<<mpInt->dT;
        // LOG(INFO)<<"                Qi "<<Qi.toRotationMatrix().eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.toRotationMatrix().eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
        // LOG(INFO)<<"             Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
         return true;
    }


Backend::Backend(double delay) : delay_(delay)
{
    thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap()
{
    map_update_.notify_one();
}

void Backend::Pause()
{
    if (status == BackendStatus::RUNNING)
    {
        std::unique_lock<std::mutex> lock(pausing_mutex_);
        status = BackendStatus::TO_PAUSE;
        pausing_.wait(lock);
    }
}

void Backend::Continue()
{
    if (status == BackendStatus::PAUSING)
    {
        status = BackendStatus::RUNNING;
        running_.notify_one();
    }
}

void Backend::BackendLoop()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(running_mutex_);
        if (status == BackendStatus::TO_PAUSE)
        {
            status = BackendStatus::PAUSING;
            pausing_.notify_one();
            running_.wait(lock);
        }
        map_update_.wait(lock);
        auto t1 = std::chrono::steady_clock::now();
        Optimize();
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Backend cost time: " << time_used.count() << " seconds.";
    }
}

void Backend::BuildProblem(Frames &active_kfs, adapt::Problem &problem,bool isimu)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;

    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        for (auto pair_feature : frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            ceres::CostFunction *cost_function;
            //if(!Imu::Num()||!initializer_->initialized)
            {
            if (first_frame->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para_kf);
            }
            else if (first_frame != frame)
            {
                double *para_fist_kf = first_frame->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(ProblemType::TwoFrameReprojectionError, cost_function, loss_function, para_fist_kf, para_kf);
            }
            }
        }
    }

    //NEWADD
    if (Imu::Num() && initializer_->initialized&&isimu)
    {
        Frame::Ptr last_frame;
        Frame::Ptr current_frame;
        bool first=true;
        //int n=active_kfs.size();
        for (auto kf_pair : active_kfs)
        {
            current_frame = kf_pair.second;
            if (!current_frame->bImu||!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
            {
                last_frame=current_frame;
               continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->Vw.data();
            auto para_bg = current_frame->ImuBias.linearized_bg.data();
            auto para_ba = current_frame->ImuBias.linearized_ba.data();
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            // if(n>10){
            //     problem.SetParameterBlockConstant(para_v);
            //     problem.SetParameterBlockConstant(para_ba);
            //     problem.SetParameterBlockConstant(para_bg);
            //     n--;
            // }

            if (last_frame && last_frame->bImu&&last_frame->last_keyframe)
            {
             if(first){
                problem.SetParameterBlockConstant(para_kf);
                first=false;
            }
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data();//恢复
                auto para_ba_last =last_frame->ImuBias.linearized_ba.data();//恢复
                ceres::CostFunction *cost_function = InertialError3::Create(current_frame->preintegration,initializer_->Rwg);
                problem.AddResidualBlock(ProblemType::IMUError,cost_function, NULL, para_kf_last, para_v_last,  para_bg_last,para_ba_last, para_kf, para_v);
                ceres::CostFunction *cost_function_g = GyroRWError3::Create(current_frame->preintegration->C.block<3,3>(9,9).inverse());
                problem.AddResidualBlock(ProblemType::IMUError,cost_function_g, NULL, para_bg_last,para_bg);
                 ceres::CostFunction *cost_function_a = AccRWError3::Create(current_frame->preintegration->C.block<3,3>(12,12).inverse());
                problem.AddResidualBlock(ProblemType::IMUError,cost_function_a, NULL, para_ba_last,para_ba);
                //LOG(INFO)<<"TIME: "<<current_frame->time-1.40364e+09+8.60223e+07<<"    V  "<<current_frame->Vw.transpose()<<"    R  "<<current_frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<current_frame->pose.translation().transpose();
               // showIMUError(para_kf_last, para_v_last,  para_bg_last,para_ba_last, para_kf, para_v,current_frame->preintegration,current_frame->time-1.40364e+09);
            }
            last_frame = current_frame;
        }
    }
    //NEWADDEND
}

double compute_reprojection_error(Vector2d ob, Vector3d pw, SE3d pose, Camera::Ptr camera)
{
    static double weights[2] = {1, 1};
    Vector2d error(0, 0);
    PoseOnlyReprojectionError(ob, pw, camera, weights)(pose.data(), error.data());
    return error.norm();
}

void Backend::Optimize()
{
    static double forward_head = 0;
    std::unique_lock<std::mutex> lock(mutex);
    // if(initializer_->initialized)
    //         std::unique_lock<std::mutex> lock2(frontend_.lock()->mutex);

    Frames active_kfs = Map::Instance().GetKeyFrames(head);
    old_pose=(--active_kfs.end())->second->pose;
    LOG(INFO)<<"BACKEND IMU OPTIMIZER  ===>"<<active_kfs.size();
    adapt::Problem problem;
    BuildProblem(active_kfs, problem);// test

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    options.max_solver_time_in_seconds = 2;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
 

/*/test
    if (Imu::Num() && initializer_->initialized){
  adapt::Problem problem_;
 ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));


  Frame::Ptr last_frame;
        Frame::Ptr current_frame;
        bool first=true;
        //int n=active_kfs.size();
        for (auto kf_pair : active_kfs)
        {
            current_frame = kf_pair.second;
            if (!current_frame->bImu||!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
            {
                last_frame=current_frame;
               continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->Vw.data();
            auto para_bg = current_frame->ImuBias.linearized_bg.data();
            auto para_ba = current_frame->ImuBias.linearized_ba.data();
            problem_.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            problem_.AddParameterBlock(para_v, 3);
            problem_.AddParameterBlock(para_ba, 3);
            problem_.AddParameterBlock(para_bg, 3);
             problem_.SetParameterBlockConstant(para_kf);
            // if(n>10){
            //     problem.SetParameterBlockConstant(para_v);
            //     problem.SetParameterBlockConstant(para_ba);
            //     problem.SetParameterBlockConstant(para_bg);
            //     n--;
            // }
            // if(first){
            //     problem.SetParameterBlockConstant(para_kf);
            //     problem.SetParameterBlockConstant(para_v);
            //     problem.SetParameterBlockConstant(para_ba);
            //     problem.SetParameterBlockConstant(para_bg);
            //     first=false;
            // }
            if (last_frame && last_frame->bImu&&last_frame->last_keyframe)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data();//恢复
                auto para_ba_last =last_frame->ImuBias.linearized_ba.data();//恢复
                ceres::CostFunction *cost_function = InertialError3::Create(current_frame->preintegration,initializer_->Rwg);
                problem_.AddResidualBlock(ProblemType::IMUError,cost_function, NULL, para_kf_last, para_v_last,  para_bg_last,para_ba_last, para_kf, para_v);
                ceres::CostFunction *cost_function_g = GyroRWError3::Create(current_frame->preintegration->C.block<3,3>(9,9).inverse());
                problem_.AddResidualBlock(ProblemType::IMUError,cost_function_g, NULL, para_bg_last,para_bg);
                 ceres::CostFunction *cost_function_a = AccRWError3::Create(current_frame->preintegration->C.block<3,3>(12,12).inverse());
                problem_.AddResidualBlock(ProblemType::IMUError,cost_function_a, NULL, para_ba_last,para_ba);
                //LOG(INFO)<<"TIME: "<<current_frame->time-1.40364e+09+8.60223e+07<<"    V  "<<current_frame->Vw.transpose()<<"    R  "<<current_frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<current_frame->pose.translation().transpose();
                // showIMUError(para_kf_last, para_v_last,  para_bg_last,para_ba_last, para_kf, para_v,current_frame->preintegration,current_frame->time-1.40364e+09);
            }
            last_frame = current_frame;
        }


    ceres::Solver::Options options_;
    options_.linear_solver_type = ceres::DENSE_SCHUR;
    options_.max_solver_time_in_seconds = 2;
    options_.num_threads = 4;
    ceres::Solver::Summary summary_;
    ceres::Solve(options_, &problem_, &summary_);
}
*/
    //NEWADD
    if(Imu::Num()&&initializer_->initialized)
    {
        for(auto kf_pair : active_kfs){
            auto frame = kf_pair.second;
            if(!frame->preintegration||!frame->last_keyframe||!frame->bImu){
                    continue;
            }
            Bias bias_(frame->ImuBias.linearized_ba[0],frame->ImuBias.linearized_ba[1],frame->ImuBias.linearized_ba[2],frame->ImuBias.linearized_bg[0],frame->ImuBias.linearized_bg[1],frame->ImuBias.linearized_bg[2]);
            frame->SetNewBias(bias_);
           // LOG(INFO)<<"opt  TIME: "<<frame->time-1.40364e+09+8.60223e+07<<"    V  "<<frame->Vw.transpose()<<"    R  "<<frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<frame->pose.translation().transpose();

        }
    }
     new_frame=(--active_kfs.end())->second;
    //NEWADDEND

    if (Lidar::Num())
    {
        mapping_->Optimize(active_kfs);
    }

    if (Navsat::Num() && Navsat::Get()->initialized)
    {
        double start_time = Navsat::Get()->Optimize((--active_kfs.end())->first);
        if (start_time && Lidar::Num())
        {
            Frames mapping_kfs = Map::Instance().GetKeyFrames(start_time);
            for (auto pair : mapping_kfs)
            {
                mapping_->ToWorld(pair.second);
            }
        }
    }

    // reject outliers and clean the map
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        auto left_features = frame->features_left;
        for (auto pair_feature : left_features)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame();
            if (compute_reprojection_error(cv2eigen(feature->keypoint), landmark->ToWorld(), frame->pose, Camera::Get()) > 10)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
            if (landmark->observations.size() == 1 && frame->id != Frame::current_frame_id)
            {
                Map::Instance().RemoveLandmark(landmark);
            }
        }
    }

    // propagate to the last frame
      forward_head = (--active_kfs.end())->first + epsilon;
    ForwardPropagate(forward_head);
       head = forward_head - delay_;


}

void Backend::ForwardPropagate(double time)
{
     std::unique_lock<std::mutex> lock(frontend_.lock()->mutex);

    Frame::Ptr last_frame = frontend_.lock()->last_frame;
    Frames active_kfs = Map::Instance().GetKeyFrames(time);
    LOG(INFO)<<"BACKEND IMU ForwardPropagate  ===>"<<active_kfs.size();
    if (active_kfs.find(last_frame->time) == active_kfs.end())
    {
        active_kfs[last_frame->time] = last_frame;
    }

    //NEWADD
    double priorA=1e5;
    double priorG=1e2;
        // if(Imu::Num() && initializer_->initialized){
        //     double dt=(--active_kfs.end())->second->time-Map::Instance().keyframes.begin()->second->time;
        //     if(dt>15&&!initA){
        //         initializer_->reinit=true;
        //         initA=true;
        //         priorA=1e5;
        //         priorG=1;
        //      //   frames_init = Map::Instance().GetKeyFrames(0,(--active_kfs.end())->second->time + epsilon,initializer_->num_frames);
        //     }
        //     else if(dt>30&&!initB){
        //        initializer_->reinit=true;
        //         initB=true;
        //         priorA=0;
        //         priorG=0;
        //       //  frames_init= Map::Instance().GetKeyFrames(0,(--active_kfs.end())->second->time + epsilon,initializer_->num_frames);
        //     }
        // }

    Frames  frames_init;
     if (Imu::Num() &&( !initializer_->initialized||initializer_->reinit))
    {
        frames_init = Map::Instance().GetKeyFrames(0,time,initializer_->num_frames);
        if (frames_init.size() == initializer_->num_frames&&frames_init.begin()->second->preintegration->isPreintegrated)
        {
            // LOG(INFO)<<frames_init.end()->second->time<<" "<<Map::Instance().keyframes.begin()->second->time;
            // if(frames_init.end()->second->time-Map::Instance().keyframes.begin()->second->time>10)
            isInitliazing=true;
        }
    }
    // bool fistinit=false;
    if (isInitliazing)
    {
        LOG(INFO)<<"Initializer Start";
        if(initializer_->InitializeIMU(frames_init,priorA,priorG))
        {
            frontend_.lock()->status = FrontendStatus::TRACKING_GOOD;
            // fistinit=true;
        }
        LOG(INFO)<<"Initializer Finished";
        isInitliazing=false;
    }    
    //     Frames  frames_init;
    //  if (Imu::Num() && !initializer_->initialized)
    // {
    //     frames_init = Map::Instance().GetKeyFrames(0,(--active_kfs.end())->second->time + epsilon,initializer_->num_frames);
    //     if (frames_init.size() == initializer_->num_frames&&frames_init.begin()->second->preintegration->isPreintegrated)
    //     {
    //         // LOG(INFO)<<frames_init.end()->second->time<<" "<<Map::Instance().keyframes.begin()->second->time;
    //         // if(frames_init.end()->second->time-Map::Instance().keyframes.begin()->second->time>10)
    //         isInitliazing=true;
    //     }
    // }


    adapt::Problem problem;
    BuildProblem(active_kfs, problem,false);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1;

    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //     if(Imu::Num()&&initializer_->initialized)
    // {
       
    //     for(auto kf_pair : active_kfs){
    //         auto frame = kf_pair.second;
    //         if(!frame->preintegration||!frame->last_keyframe||!frame->bImu){
    //                 continue;
    //         }
    //         Bias bias_(frame->ImuBias.linearized_ba[0],frame->ImuBias.linearized_ba[1],frame->ImuBias.linearized_ba[2],frame->ImuBias.linearized_bg[0],frame->ImuBias.linearized_bg[1],frame->ImuBias.linearized_bg[2]);
    //         frame->SetNewBias(bias_);
    //         //LOG(INFO)<<"fwd  TIME: "<<frame->time-1.40364e+09+8.60223e+07<<"    V  "<<frame->Vw.transpose()<<"    R  "<<frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<frame->pose.translation().transpose();
    //     }
    // }
    if(Imu::Num() && initializer_->initialized)
    {

        Frame::Ptr last_key_frame=new_frame;
        for(auto kf:active_kfs){
            Frame::Ptr current_key_frame =kf.second;
            // SE3d transFromOld=current_key_frame->pose*old_pose.inverse();
            // current_key_frame->pose=transFromOld*new_frame->pose;
            Vector3d Gz ;
            Gz << 0, 0, -9.81007;
            double t12=current_key_frame->preintegration->dT;
            Vector3d twb1=last_key_frame->GetImuPosition();
            Matrix3d Rwb1=last_key_frame->GetImuRotation();
            Vector3d Vwb1=last_key_frame->Vw;
            // Matrix3d Rwb2=NormalizeRotation(Rwb1*current_key_frame->preintegration->GetDeltaRotation(last_key_frame->GetImuBias()));
            // Vector3d twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_key_frame->preintegration->GetDeltaPosition(last_key_frame->GetImuBias());
            Vector3d Vwb2=Vwb1+t12*Gz+Rwb1*current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias());
            current_key_frame->SetVelocity(Vwb2);
             //current_key_frame->SetPose(Rwb2,twb2);
            current_key_frame->SetNewBias(last_key_frame->GetImuBias());
            last_key_frame=current_key_frame;
//         //              LOG(INFO)<<"ForwardPropagate  "<<current_key_frame->time-1.40364e+09<<"  T12  "<<t12;
//         //             Matrix3d tcb;
//         //     tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
//         //     0.999557249008, 0.0149672133247, 0.025715529948,
//         //     -0.0257744366974, 0.00375618835797, 0.999660727178;
//         //             LOG(INFO)<<" Rwb2\n"<<tcb.inverse()*Rwb2;
//         // LOG(INFO)<<" Vwb1"<<Vwb1.transpose();
//         // LOG(INFO)<<"   GRdV "<<(t12*Gz+Rwb1*current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias())).transpose()*tcb;
//         // LOG(INFO)<<"   RdV    "<<((tcb.inverse()*Rwb1)*current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias())).transpose();
//         // LOG(INFO)<<"   dV      "<<(current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias())).transpose();
                 adapt::Problem problem;
      ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
          ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
      auto para_kf=current_key_frame->pose.data();
      problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
      /* test
      for (auto pair_feature : current_key_frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            ceres::CostFunction *cost_function;
            cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint),landmark->ToWorld(),  Camera::Get(), current_key_frame->weights.visual);
            problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para_kf);
        }
*/
       if(current_key_frame->last_keyframe&&initializer_->initialized&&current_key_frame->preintegration->isPreintegrated)
        {
            auto para_kf_last=current_key_frame->last_keyframe->pose.data();
            auto para_v=current_key_frame->Vw.data();
            auto para_v_last=current_key_frame->last_keyframe->Vw.data();
            auto para_accBias=current_key_frame->ImuBias.linearized_ba.data();
            auto para_gyroBias=current_key_frame->ImuBias.linearized_bg.data();
            auto para_accBias_last=current_key_frame->last_keyframe->ImuBias.linearized_ba.data();
            auto para_gyroBias_last=current_key_frame->last_keyframe->ImuBias.linearized_bg.data();
            problem.AddParameterBlock(para_kf_last, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_v_last, 3);
            problem.AddParameterBlock(para_accBias, 3);
            problem.AddParameterBlock(para_gyroBias, 3);
            problem.AddParameterBlock(para_accBias_last, 3);
            problem.AddParameterBlock(para_gyroBias_last, 3);
            problem.SetParameterBlockConstant(para_kf_last);
            problem.SetParameterBlockConstant(para_kf);
            problem.SetParameterBlockConstant(para_accBias_last);
            problem.SetParameterBlockConstant(para_gyroBias_last);
             ceres::CostFunction *cost_function =InertialError3::Create(current_key_frame->preintegration,initializer_->Rwg);
            problem.AddResidualBlock(ProblemType::IMUError,cost_function, NULL, para_kf_last, para_v_last,  para_gyroBias,para_accBias, para_kf, para_v);
            ceres::CostFunction *cost_function_g = GyroRWError3::Create(current_key_frame->preintegration->C.block<3,3>(9,9).inverse());
            problem.AddResidualBlock(ProblemType::IMUError,cost_function_g, NULL, para_gyroBias_last,para_gyroBias);
            ceres::CostFunction *cost_function_a = AccRWError3::Create(current_key_frame->preintegration->C.block<3,3>(12,12).inverse());
            problem.AddResidualBlock(ProblemType::IMUError,cost_function_a, NULL, para_accBias_last,para_accBias);
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_solver_time_in_seconds = 0.01;
    //options.max_num_iterations = 4;
        options.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);


        current_key_frame->SetNewBias(current_key_frame->ImuBias);
         LOG(INFO)<<"Forward  "<<current_key_frame->time-1.40364e+09+8.60223e+07<<"  T12  "<<t12<<"  Vwb1  "<<Vwb1.transpose()<<" Vwb2  "<<Vwb2.transpose();
        // LOG(INFO)<<"   GRdV   "<<(t12*Gz+Rwb1*current_frame->preintegrationFrame->GetDeltaVelocity(last_frame->GetImuBias())).transpose();
        LOG(INFO)<<"   RdV    "<<((Rwb1)*current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias())).transpose()<<"   dV      "<<(current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias())).transpose();
        LOG(INFO)<<"    BIAS   a "<<last_key_frame->GetImuBias().linearized_ba.transpose()<<" g "<<last_key_frame->GetImuBias().linearized_bg.transpose();
        }
    }


   if(Imu::Num()&&initializer_->initialized)
   {
 /* test
          adapt::Problem problem_;
    BuildProblem(active_kfs, problem_);

    ceres::Solver::Options options_;
    options_.linear_solver_type = ceres::DENSE_SCHUR;
    options_.max_num_iterations = 1;
    options_.num_threads = 4;
    ceres::Solver::Summary summary_;
    ceres::Solve(options_, &problem_, &summary_);
        if(Imu::Num()&&initializer_->initialized)
    {
       
        for(auto kf_pair : active_kfs){
            auto frame = kf_pair.second;
            if(!frame->preintegration||!frame->last_keyframe||!frame->bImu){
                    continue;
            }
            Bias bias_(frame->ImuBias.linearized_ba[0],frame->ImuBias.linearized_ba[1],frame->ImuBias.linearized_ba[2],frame->ImuBias.linearized_bg[0],frame->ImuBias.linearized_bg[1],frame->ImuBias.linearized_bg[2]);
            frame->SetNewBias(bias_);
            //LOG(INFO)<<"fwd  TIME: "<<frame->time-1.40364e+09+8.60223e+07<<"    V  "<<frame->Vw.transpose()<<"    R  "<<frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<frame->pose.translation().transpose();
        }
    }
*/

       Map::Instance().mapUpdated=false;
         frontend_.lock()->UpdateFrameIMU((--active_kfs.end())->second->GetImuBias());
   }
    //NEWADDEND

    frontend_.lock()->UpdateCache();

}

// NEWADD


// Vector3d Backend::ComputeGyroBias(const Frames &frames)
// {
//     const int N = frames.size();
//     std::vector<double> vbx;
//     vbx.reserve(N);
//     std::vector<double> vby;
//     vby.reserve(N);
//     std::vector<double> vbz;
//     vbz.reserve(N);

//     Matrix3d H = Matrix3d::Zero();
//     Vector3d grad = Vector3d::Zero();
//     bool first=true;
//     Frame::Ptr pF1;
//     for(auto iter:frames)
//     {
//         if(first){
//             pF1=iter.second;
//            first=false; 
//             continue;
//         }
//         Frame::Ptr pF2 = iter.second;
//         Matrix3d VisionR = pF1->GetImuRotation().transpose()*pF2->GetImuRotation();
//         Matrix3d JRg = pF2->preintegration->JRg;
//         Matrix3d E = pF2->preintegrationFrame->GetUpdatedDeltaRotation().transpose()*VisionR;
//         Vector3d e =  LogSO3(E);
//         assert(fabs(pF2->time-pF1->time-pF2->preintegration->dT)<0.01);

//        Matrix3d J = -InverseRightJacobianSO3(e)*E.transpose()*JRg;
//         grad += J.transpose()*e;
//         H += J.transpose()*J;
//         pF1=iter.second;
//     }

//     Vector3d bg = -H.inverse()*grad;

//   for(auto iter:frames)
//     {
//         Frame::Ptr pF = iter.second;
//         pF->SetNewBias();
//     }
//     return bg;
// }

// Vector3d Backend::ComputeVelocitiesAccBias(const Frames &frames)
// {
//     const int N = frames.size();
//     const int nVar = 3*N +3; // 3 velocities/frame + acc bias
//     const int nEqs = 6*(N-1);

//     cv::Mat J(nEqs,nVar,CV_32F,cv::Scalar(0));
//     cv::Mat e(nEqs,1,CV_32F,cv::Scalar(0));
//     cv::Mat g = (cv::Mat_<float>(3,1)<<0,0,-G);

//     for(int i=0;i<N-1;i++)
//     {
//         Frame* pF2 = vpFs[i+1];
//         Frame* pF1 = vpFs[i];
//         cv::Mat twb1 = pF1->GetImuPosition();
//         cv::Mat twb2 = pF2->GetImuPosition();
//         cv::Mat Rwb1 = pF1->GetImuRotation();
//         cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
//         cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
//         cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
//         cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
//         float t12 = pF2->mpImuPreintegratedFrame->dT;
//         // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
//         J.rowRange(6*i,6*i+3).colRange(3*i,3*i+3) += cv::Mat::eye(3,3,CV_32F)*t12;
//         J.rowRange(6*i,6*i+3).colRange(3*N,3*N+3) += Rwb1*JP12;
//         e.rowRange(6*i,6*i+3) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
//         // Velocity v2=v1+g*t+R1*dV12
//         J.rowRange(6*i+3,6*i+6).colRange(3*i,3*i+3) += -cv::Mat::eye(3,3,CV_32F);
//         J.rowRange(6*i+3,6*i+6).colRange(3*(i+1),3*(i+1)+3) += cv::Mat::eye(3,3,CV_32F);
//         J.rowRange(6*i+3,6*i+6).colRange(3*N,3*N+3) -= Rwb1*JV12;
//         e.rowRange(6*i+3,6*i+6) = g*t12+Rwb1*dV12;
//     }

//     cv::Mat H = J.t()*J;
//     cv::Mat B = J.t()*e;
//     cv::Mat x(nVar,1,CV_32F);
//     cv::solve(H,B,x);

//     bax = x.at<float>(3*N);
//     bay = x.at<float>(3*N+1);
//     baz = x.at<float>(3*N+2);

//     for(int i=0;i<N;i++)
//     {
//         Frame* pF = vpFs[i];
//         x.rowRange(3*i,3*i+3).copyTo(pF->mVw);
//         if(i>0)
//         {
//             pF->mImuBias.bax = bax;
//             pF->mImuBias.bay = bay;
//             pF->mImuBias.baz = baz;
//             pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
//         }
//     }
// }


//NEWADDEND
} // namespace lvio_fusion