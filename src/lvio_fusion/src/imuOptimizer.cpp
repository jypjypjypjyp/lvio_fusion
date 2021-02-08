#include "lvio_fusion/imu/imuOptimizer.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/imu_error.hpp"

namespace lvio_fusion
{
    void  ImuOptimizer::ComputeGyroBias(const Frames &frames)
    {
        const int N = frames.size();
        std::vector<double> vbx;
        vbx.reserve(N);
        std::vector<double> vby;
        vby.reserve(N);
        std::vector<double> vbz;
        vbz.reserve(N);

        Matrix3d H = Matrix3d::Zero();
        Vector3d grad = Vector3d::Zero();
        bool first=true;
        Frame::Ptr pF1;
        for(auto iter:frames)
        {
            if(first){
                pF1=iter.second;
            first=false; 
                continue;
            }
            Frame::Ptr pF2 = iter.second;
            Matrix3d VisionR = pF1->GetImuRotation().transpose()*pF2->GetImuRotation();
            Matrix3d JRg = pF2->preintegration-> jacobian.block<3, 3>(3, 12);
            Matrix3d E = pF2->preintegrationFrame->GetUpdatedDeltaRotation().inverse().toRotationMatrix()*VisionR;
            Vector3d e =  LogSO3(E);
        // assert(fabs(pF2->time-pF1->time-pF2->preintegration->dT)<0.01);

        Matrix3d J = -InverseRightJacobianSO3(e)*E.transpose()*JRg;
            grad += J.transpose()*e;
            H += J.transpose()*J;
            pF1=iter.second;
        }

        Vector3d bg = -H.ldlt().solve( grad);
        LOG(INFO)<<bg.transpose();
        for(auto iter:frames)
            {
                Frame::Ptr pF = iter.second;
                pF->ImuBias.linearized_ba=Vector3d::Zero();
                pF->ImuBias.linearized_bg=bg;
                pF->SetNewBias(pF->ImuBias);
            }
        return;
    }

    void  ImuOptimizer::ComputeVelocitiesAccBias(const Frames &frames)
    {
        const int N = frames.size();
        const int nVar = 3*N +3; // 3 velocities/frame + acc bias
        const int nEqs = 6*(N-1);

        MatrixXd J(nEqs,nVar);
        J.setZero();
        VectorXd e(nEqs);
        e.setZero();
        Vector3d g;
        g<<0,0,-9.81007;

        int i=0;
        bool first=true;
        Frame::Ptr Frame1;
        Frame::Ptr Frame2;
        for(auto iter:frames)
        {
            if(first){
                Frame1=iter.second;
                first=false;
                continue;
            }

            Frame2=iter.second;
            Vector3d twb1 = Frame1->GetImuPosition();
            Vector3d twb2 = Frame2->GetImuPosition();
            Matrix3d Rwb1 = Frame1->GetImuRotation();
            Vector3d dP12 = Frame2->preintegration->GetUpdatedDeltaPosition();
            Vector3d dV12 = Frame2->preintegration->GetUpdatedDeltaVelocity();
            Matrix3d JP12 = Frame2->preintegration-> jacobian.block<3, 3>(0, 9);
            Matrix3d JV12 = Frame2->preintegration->jacobian.block<3, 3>(6, 9);
            float t12 = Frame2->preintegration->sum_dt;
            // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
            J.block<3,3>(6*i,3*i)+= Matrix3d::Identity()*t12;
            J.block<3,3>(6*i,3*N) += Rwb1*JP12;
            e.block<3,1>(6*i,0) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
            // Velocity v2=v1+g*t+R1*dV12
            J.block<3,3>(6*i+3,3*i)+= -Matrix3d::Identity();
            J.block<3,3>(6*i+3,3*i+3)+=  Matrix3d::Identity();
            J.block<3,3>(6*i+3,3*N) -= Rwb1*JV12;
            e.block<3,1>(6*i+3,0) = g*t12+Rwb1*dV12;
            Frame1=Frame2;
            i++;
        }

        MatrixXd H = J.transpose()*J;
        MatrixXd B = J.transpose()*e;
        VectorXd x=H.ldlt().solve(B);
        Vector3d ba;
        ba(0) = x(3*N);
        ba(1) = x(3*N+1);
        ba(2) = x(3*N+2);
    
        i=0;
        for(auto iter:frames)
        {
            Frame::Ptr pF = iter.second;
            pF->preintegration->Repropagate(ba, pF->ImuBias.linearized_bg);
            pF->Vw=x.block<3,1>(3*i,0);
            if(i>0)
            {
                pF->ImuBias.linearized_ba=ba;
                pF->SetNewBias(pF->ImuBias);
            }
            i++;
        }
        return;
    }
    
     void  ImuOptimizer::ReComputeBiasVel( Frames &frames,Frame::Ptr &prior_frame )
    {
        adapt::Problem problem;
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));


        Frame::Ptr last_frame=prior_frame;
        Frame::Ptr current_frame;
        bool first=true;
        //int n=active_kfs.size();
        if(frames.size()>0)
        for (auto kf_pair : frames)
        {
            current_frame = kf_pair.second;
            if (!current_frame->bImu||!current_frame->last_keyframe||current_frame->preintegration==nullptr)
            {
                last_frame=current_frame;
               continue;
            }
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->Vw.data();
            auto para_bg = current_frame->ImuBias.linearized_bg.data();
            auto para_ba = current_frame->ImuBias.linearized_ba.data();
            problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            problem.SetParameterBlockConstant(para_kf);
            if (last_frame && last_frame->bImu&&last_frame->last_keyframe)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->Vw.data();
                auto para_bg_last = last_frame->ImuBias.linearized_bg.data();//恢复
                auto para_ba_last =last_frame->ImuBias.linearized_ba.data();//恢复
                if(first){
                    problem.AddParameterBlock(para_kf_last, SE3d::num_parameters, local_parameterization);
                    problem.AddParameterBlock(para_v_last, 3);
                    problem.AddParameterBlock(para_bg_last, 3);
                    problem.AddParameterBlock(para_ba_last, 3);
                    problem.SetParameterBlockConstant(para_kf_last);
                    problem.SetParameterBlockConstant(para_v_last);
                    problem.SetParameterBlockConstant(para_bg_last);
                    problem.SetParameterBlockConstant(para_ba_last);
                    first=false;
                }
                ceres::CostFunction *cost_function = ImuError::Create(current_frame->preintegration);
                problem.AddResidualBlock(ProblemType::IMUError,cost_function, NULL, para_kf_last, para_v_last,para_ba_last,para_bg_last, para_kf, para_v,para_ba,para_bg);
                //showIMUError(para_kf_last, para_v_last,para_ba_last,para_bg_last, para_kf, para_v,para_ba,para_bg,current_frame->preintegration,current_frame->time-1.40364e+09+8.60223e+07);

            }
            last_frame = current_frame;
        }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations =4;
    options.max_solver_time_in_seconds=0.1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
     LOG(INFO)<<"ReComputeBiasVel  "<<summary.BriefReport();
       for(auto kf_pair : frames){
            auto frame = kf_pair.second;
            if(!frame->preintegration||!frame->last_keyframe||!frame->bImu){
                    continue;
            }
            Bias bias_(frame->ImuBias.linearized_ba[0],frame->ImuBias.linearized_ba[1],frame->ImuBias.linearized_ba[2],frame->ImuBias.linearized_bg[0],frame->ImuBias.linearized_bg[1],frame->ImuBias.linearized_bg[2]);
            frame->SetNewBias(bias_);
           // LOG(INFO)<<"opt  TIME: "<<frame->time-1.40364e+09+8.60223e+07<<"    V  "<<frame->Vw.transpose()<<"    R  "<<frame->pose.rotationMatrix().eulerAngles(0,1,2).transpose()<<"    P  "<<frame->pose.translation().transpose();
        }
        return ;
    }

    void  ImuOptimizer::RePredictVel(Frames &frames,Frame::Ptr &prior_frame )
    {
           Frame::Ptr last_key_frame=prior_frame;
        bool first=true;
        for(auto kf:frames){
            Frame::Ptr current_key_frame =kf.second;
            Vector3d Gz ;
            Gz << 0, 0, -9.81007;
            double t12=current_key_frame->preintegration->sum_dt;
            Vector3d twb1=last_key_frame->GetImuPosition();
            Matrix3d Rwb1=last_key_frame->GetImuRotation();
            Vector3d Vwb1;
            Vwb1=last_key_frame->Vw;

            Matrix3d Rwb2=NormalizeRotation(Rwb1*current_key_frame->preintegration->GetDeltaRotation(last_key_frame->GetImuBias()).toRotationMatrix());
            Vector3d twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_key_frame->preintegration->GetDeltaPosition(last_key_frame->GetImuBias());
            Vector3d Vwb2=Vwb1+t12*Gz+Rwb1*current_key_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias());
            current_key_frame->SetVelocity(Vwb2);
            current_key_frame->SetNewBias(last_key_frame->GetImuBias());
            last_key_frame=current_key_frame;
        }
    }


} // namespace lvio_fusioni
