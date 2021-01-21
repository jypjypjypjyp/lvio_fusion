#include "lvio_fusion/imu/initializer.h"
#include <lvio_fusion/utility.h>
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
//#include "lvio_fusion/optimizer.h"
#include <math.h>
#include <ceres/ceres.h>
namespace lvio_fusion
{
 bool showGSIMU(const  double *  parameters0,const  double *  parameters1,const  double *  parameters2,const  double *  parameters3,const  double *  parameters4,imu::Preintegration::Ptr mpInt,SE3d current_pose,SE3d last_pose)  
    {
    //    Quaterniond Qi(last_pose.rotationMatrix());
       Matrix3d Qi=last_pose.rotationMatrix();
        Vector3d Pi=last_pose.translation();
        Vector3d Vi(parameters0[0], parameters0[1], parameters0[2]);
       Matrix3d Qj=current_pose.rotationMatrix();
        // Quaterniond Qj(current_pose.rotationMatrix());
        Vector3d Pj=current_pose.translation();
        Vector3d Vj(parameters1[0], parameters1[1], parameters1[2]);
        Vector3d gyroBias(parameters2[0], parameters2[1], parameters2[2]);
        Vector3d accBias(parameters3[0], parameters3[1], parameters3[2]);
        Quaterniond rwg(parameters4[3],parameters4[0], parameters4[1], parameters4[2]);
        // double Scale=parameters[5][0];
        double Scale=1.0;
        double dt=(mpInt->dT);
        // Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(0),Vector3d::UnitX()));
        // Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1),Vector3d::UnitY()));
        // Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(2),Vector3d::UnitZ()));
        Matrix3d Rwg=rwg.toRotationMatrix();
        LOG(INFO)<<Rwg;
        // Rwg= yawAngle*pitchAngle*rollAngle;
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
        Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
        // LOG(INFO)<<"G  "<<(g).transpose()<<"Rwg"<<parameters4[0]<<" "<< parameters4[1]<<" "<< parameters4[2]<<" "<<parameters4[3];
        const Bias  b1(accBias(0,0),accBias(1,0),accBias(2,0),gyroBias(0,0),gyroBias(1,0),gyroBias(2,0));
        Matrix3d dR = (mpInt->GetDeltaRotation(b1));
        Vector3d dV = (mpInt->GetDeltaVelocity(b1));
        Vector3d dP =(mpInt->GetDeltaPosition(b1));
        Quaterniond corrected_delta_q(dR);
       const Vector3d er=  LogSO3(dR.transpose()*Qi.transpose()*Qj);
        // const Vector3d er =  2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        const Vector3d ev = Qi.inverse()*(Scale*(Vj - Vi) - g*dt) - dV;
        const Vector3d ep = Qi.inverse()*(Scale*(Pj - Pi - Vi*dt) - g*dt*dt/2) - dP;
        
  Matrix<double, 9, 1> residual;
        residual<<er,ev,ep;
    //    LOG(INFO)<<"\nInertialGSError residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
    //            LOG(INFO)<<"\ndV "<<dV.transpose()<< "  dP "<<dP.transpose()<<"\ndR\n"<<dR;
    //     LOG(INFO)<<"\nQi"<<tcb.inverse()*Qi;
    //        Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
    //      Info = (Info+Info.transpose())/2;
    //      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    //      Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
    //      for(int i=0;i<9;i++)
    //          if(eigs[i]<1e-12)
    //              eigs[i]=0;
    //      Matrix<double, 9,9> sqrt_info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
       Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>(mpInt->C.block<9,9>(0,0).inverse()).matrixL().transpose();
        sqrt_info/=InfoScale;
      //  assert(residual[0]<10&&residual[1]<10&&residual[2]<10&&residual[3]<10&&residual[4]<10&&residual[5]<10&&residual[6]<10&&residual[7]<10&&residual[8]<10);
        residual = sqrt_info* residual;
    //    LOG(INFO)<<"InertialGSError sqrt_info* residual :  er "<<residual.transpose()<<" dT "<<mpInt->dT;
    //     LOG(INFO)<<"                Qi "<<Qi.eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
    //     LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
    //     LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
    //      LOG(INFO)<<"                 Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
        return true;
    }


Bias InertialOptimization(Frames &key_frames, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, double priorG, double priorA)   
{
    ceres::Problem problem;

    //先验BIAS约束
    auto para_gyroBias=key_frames.begin()->second->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);
    ceres::CostFunction *cost_function = PriorGyroError3::Create(Vector3d::Zero(),priorG);
    problem.AddResidualBlock(cost_function,NULL,para_gyroBias);

    auto para_accBias=key_frames.begin()->second->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);
    cost_function = PriorAccError3::Create(Vector3d::Zero(),priorA);
    problem.AddResidualBlock(cost_function,NULL,para_accBias);
    
    //优化重力、BIAS和速度的边
    Quaterniond rwg(Rwg);
    LOG(INFO)<<rwg.w()<<" "<<rwg.x()<<" "<<rwg.y()<<" "<<rwg.z();
    SO3d RwgSO3(rwg);
LOG(INFO)<<rwg.toRotationMatrix();
    auto para_rwg=RwgSO3.data();
    //LOG(INFO)<<para_rwg[0]<<" "<<para_rwg[1]<<" "<<para_rwg[2]<<" "<<para_rwg[3];

        ceres::LocalParameterization *local_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(para_rwg, SO3d::num_parameters,local_parameterization);
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    double *para_scale=&scale;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if (!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_v = current_frame->Vw.data();
        problem.AddParameterBlock(para_v, 3);   

        if (last_frame)
        {
            auto para_v_last = last_frame->Vw.data();
            cost_function = InertialGSError3::Create(current_frame->preintegration,current_frame->pose,last_frame->pose);
            problem.AddResidualBlock(cost_function, NULL,para_v_last,  para_v,para_gyroBias,para_accBias,para_rwg);
           // LOG(INFO)<<current_frame->preintegration->dV.transpose();

            // showGSIMU(para_v_last,  para_v,para_gyroBias,para_accBias,para_rwg,current_frame->preintegration,current_frame->pose,last_frame->pose);
                // LOG(INFO)<<"InertialOptimization ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id<<"  t"<<current_frame->preintegration->dT;
                // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose(); 
                // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
                // LOG(INFO)<<"     current velocity: "<<current_frame->Vw.transpose();
                // LOG(INFO)<<"     last  velocity: "<<last_frame->Vw.transpose();
        }
        last_frame = current_frame;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.1;
    // options.max_num_iterations =1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::this_thread::sleep_for(std::chrono::seconds(3));

    //数据恢复
    // Eigen::AngleAxisd rollAngle(AngleAxisd(rwg(0),Vector3d::UnitX()));
    // Eigen::AngleAxisd pitchAngle(AngleAxisd(rwg(1),Vector3d::UnitY()));
    // Eigen::AngleAxisd yawAngle(AngleAxisd(rwg(2),Vector3d::UnitZ()));
    // Rwg= yawAngle*pitchAngle*rollAngle;
    Quaterniond rwg2(RwgSO3.data()[3],RwgSO3.data()[0],RwgSO3.data()[1],RwgSO3.data()[2]);
        Rwg=rwg2.toRotationMatrix();
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
                        Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
    //   LOG(INFO)<<"OPTG "<<(g).transpose()<<"Rwg"<<RwgSO3.data()[0]<<" "<<RwgSO3.data()[1]<<" "<<RwgSO3.data()[2]<<" "<<RwgSO3.data()[3];
    Bias bias_(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
    bg << para_gyroBias[0],para_gyroBias[1],para_gyroBias[2];
    ba <<para_accBias[0],para_accBias[1],para_accBias[2];
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        Vector3d dbg=current_frame->GetGyroBias()-bg;
        if(dbg.norm() >0.01)
        {
            current_frame->SetNewBias(bias_);
           // current_frame->preintegration->Reintegrate();
        }
       else
        {
            current_frame->SetNewBias(bias_);
        }     
            Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
        LOG(INFO)<<current_frame->preintegration->dV.transpose();
        LOG(INFO)<<"InertialOptimization  "<<current_frame->time-1.40364e+09/*+8.60223e+07*/<<"   Vwb1  "<<current_frame->Vw.transpose()*tcb;//<<"\nR: \n"<<tcb.inverse()*current_frame->pose.rotationMatrix();
    }
    LOG(INFO)<<"BIAS   a "<<bias_.linearized_ba.transpose()<<" g "<<bias_.linearized_bg.transpose();
    
    return bias_;
}
     bool showIMUErrorINIT(const double*  parameters0, const double*  parameters1, const double*  parameters2, const double*  parameters3, const double*  parameters4, const double*  parameters5, imu::Preintegration::Ptr mpInt, double time)  
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
                    Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
        LOG(INFO)<<"\nInertialError residual "<<residual.transpose();
        LOG(INFO)<<"\ndV "<<dV.transpose()<< "  dP "<<dP.transpose()<<"\ndR\n"<<dR;
        LOG(INFO)<<"\nVi"<<(tcb.inverse()*Vi).transpose()<<"Pi"<<(tcb.inverse()*Pi).transpose()<<"\nQi\n"<<tcb.inverse()*Qi.toRotationMatrix();
        //    Matrix<double ,9,9> Info=mpInt->C.block<9,9>(0,0).inverse();
        //  Info = (Info+Info.transpose())/2;
        //  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
        //  Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
        //  for(int i=0;i<9;i++)
        //      if(eigs[i]<1e-12)
        //          eigs[i]=0;
        //  Matrix<double, 9,9> sqrt_info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
        Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>( mpInt->C.block<9,9>(0,0).inverse()).matrixL().transpose();
        sqrt_info/=InfoScale;
        // LOG(INFO)<<"InertialError sqrt_info "<<sqrt_info;
        //assert(!isnan(residual[0])&&!isnan(residual[1])&&!isnan(residual[2])&&!isnan(residual[3])&&!isnan(residual[4])&&!isnan(residual[5])&&!isnan(residual[6])&&!isnan(residual[7])&&!isnan(residual[8]));
        residual = sqrt_info* residual;
        // LOG(INFO)<<time<<"  IMUError:  r "<<residual.transpose()<<"  "<<mpInt->dT;
        // LOG(INFO)<<"                Qi "<<Qi.toRotationMatrix().eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.toRotationMatrix().eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
        // LOG(INFO)<<"                Vi "<<Vi.transpose()<<" Vj "<<Vj.transpose()<<"dV"<<dV.transpose();
        // LOG(INFO)<<"             Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
         return true;
    }

Bias FullInertialBA(Frames &key_frames, Matrix3d Rwg, double priorG, double priorA)
{
    ceres::Problem problem;
    
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double start_time = key_frames.begin()->first;
    for (auto pair_kf : key_frames)
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
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
        }
    }

    Frame::Ptr pIncKF=key_frames.begin()->second;
    double* para_gyroBias;
    double* para_accBias;
    para_gyroBias=pIncKF->ImuBias.linearized_bg.data();
    problem.AddParameterBlock(para_gyroBias, 3);   
    para_accBias=pIncKF->ImuBias.linearized_ba.data();
    problem.AddParameterBlock(para_accBias, 3);   

    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    int i=key_frames.size();
    for (auto kf_pair : key_frames)
    {
        i--;
        current_frame = kf_pair.second;
        if (!current_frame->bImu||!current_frame->last_keyframe||!current_frame->preintegration->isPreintegrated)
        {
            last_frame=current_frame;
            continue;
        }
        auto para_kf = current_frame->pose.data();
        auto para_v = current_frame->Vw.data();
        //problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        problem.AddParameterBlock(para_v, 3);
        if (last_frame && last_frame->bImu)
        {
            auto para_kf_last = last_frame->pose.data();
            auto para_v_last = last_frame->Vw.data();
            ceres::CostFunction *cost_function =InertialError3::Create(current_frame->preintegration,Rwg);
            problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last,  para_gyroBias,para_accBias, para_kf, para_v);
            // showIMUErrorINIT(para_kf_last, para_v_last,  para_gyroBias,para_accBias, para_kf, para_v,current_frame->preintegration,current_frame->time-1.40364e+09);
            //LOG(INFO)<<current_frame->preintegration->dV.transpose();
                // LOG(INFO)<<"FullInertialBA ckf: "<<current_frame->id<<"  lkf: "<<last_frame->id;
                // LOG(INFO)<<"     current pose: "<<current_frame->pose.translation().transpose(); 
                // LOG(INFO)<<"     last pose: "<<last_frame->pose.translation().transpose();
                // LOG(INFO)<<"     current velocity: "<<current_frame->Vw.transpose();
                // LOG(INFO)<<"     last  velocity: "<<last_frame->Vw.transpose();
        }
        last_frame = current_frame;
    }
    //epa  para_accBias
    ceres::CostFunction *cost_function = PriorAccError3::Create(Vector3d::Zero(),priorA);
    problem.AddResidualBlock(cost_function, NULL, para_accBias);
    //epg para_gyroBias
    cost_function = PriorGyroError3::Create(Vector3d::Zero(),priorG);
    problem.AddResidualBlock(cost_function, NULL, para_gyroBias);
    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    //options.max_solver_time_in_seconds =0.01;
    options.max_num_iterations=4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    summary.FullReport();
    //数据恢复
    Bias lastBias;
    for(Frames::iterator iter = key_frames.begin(); iter != key_frames.end(); iter++)
    {
        current_frame=iter->second;
        if(current_frame->bImu&&current_frame->last_keyframe)
        {
            Bias bias(para_accBias[0],para_accBias[1],para_accBias[2],para_gyroBias[0],para_gyroBias[1],para_gyroBias[2]);
            current_frame->SetNewBias(bias);
            lastBias=bias;
               Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
             LOG(INFO)<<"FullInertialBA  "<<current_frame->time-1.40364e+09/*+8.60223e+07*/<<"   Vwb1  "<<current_frame->Vw.transpose()*tcb;//<<"\nR: \n"<<tcb.inverse()*current_frame->pose.rotationMatrix();
        }
    }
    LOG(INFO)<<"BIAS   a "<<lastBias.linearized_ba.transpose()<<" g "<<lastBias.linearized_bg.transpose();
    return lastBias;
}

bool  Initializer::estimate_Vel_Rwg(std::vector< Frame::Ptr > Key_frames)
{
     std::vector<double>  times;
        std::vector<Vector3d>  Ps;
        std::vector<Matrix3d>  Rs;

        std::vector<Vector3d>  dVs;
        std::vector<Vector3d>  dPs;
        std::vector<Matrix3d>  dRs;
        std::vector<Matrix<double,15,15>>  Cs;

        std::vector<Matrix3d>  JRgs;
        std::vector<Matrix3d>  JVgs;
        std::vector<Matrix3d>  JVas;
        std::vector<Matrix3d>  JPgs;
        std::vector<Matrix3d>  JPas;

        
        times.reserve(10);
        Ps.reserve(10);
        Rs.reserve(10);
       
        dVs.reserve(10);
        dPs.reserve(10);
        dRs.reserve(10);
        Cs.reserve(10);

        JRgs.reserve(10);
        JVgs.reserve(10);
        JVas.reserve(10);
        JPgs.reserve(10);
        JPas.reserve(10);


        times[0]=-3371.0364444255829+1.40364e+09;
        times[1]=-3370.7864444255829+1.40364e+09;
        times[2]=-3370.5364444255829+1.40364e+09;
        times[3]=-3370.2864444255829+1.40364e+09;
        times[4]=-3370.0364444255829+1.40364e+09;
        times[5]=-3369.7864444255829 +1.40364e+09;
        times[6]=-3369.5364444255829+1.40364e+09;
        times[7]=-3369.2864444255829+1.40364e+09;
        times[8]=-3369.0364444255829+1.40364e+09;
        times[9]=-3368.7864444255829+1.40364e+09;

        Ps[0]<<0.065222919, -0.020706384, -0.0080546029;
        Ps[1]<<0.13616742, -0.058817036, 0.0072276667;
        Ps[2]<<0.20901129, -0.080879375, 0.049652021;
        Ps[3]<<0.29620379, -0.06541203, 0.10217573;
        Ps[4]<<0.40279835, -0.031201176, 0.1734584;
        Ps[5]<<0.51777601, -0.043059722, 0.25133854;
        Ps[6]<<0.63387346, -0.098639309, 0.33116522;
        Ps[7]<<0.74413091, -0.17301267, 0.40232551;
        Ps[8]<<0.8524515, -0.21914294, 0.48539206;
        Ps[9]<<0.96178234, -0.24136597, 0.57736999;

        Rs[0]<<0.014865543, 0.99955726, -0.025774436,
 -0.99988091, 0.014967213, 0.0037561883,
 0.004140297, 0.02571553, 0.99966073;
        Rs[1]<<0.02330914, 0.99966568, 0.011186309,
 -0.99964893, 0.023164896, 0.012855336,
 0.012591908, -0.011482023, 0.9998548;
         Rs[2]<<0.060608804, 0.99696088, 0.048944756,
 -0.99803007, 0.061323836, -0.013240544,
 -0.016201781, -0.048045844, 0.99871373;
         Rs[3]<<0.09429688, 0.98763293, 0.125257,
 -0.99554151, 0.093261078, 0.014121025,
 0.0022647865, -0.12603012, 0.99202383;
         Rs[4]<<0.1250338, 0.9658913, 0.22676101,
 -0.99195409, 0.11713015, 0.048036538,
 0.019837528, -0.23094268, 0.97276509;
         Rs[5]<<0.12968302, 0.95051467, 0.28231922,
 -0.99123859, 0.13147354, 0.012678154,
 -0.025066735, -0.28148985, 0.95923674;
         Rs[6]<<0.12596861, 0.94638234, 0.29747662,
 -0.99144405, 0.13044167, 0.0048512854,
 -0.034212172, -0.29554254, 0.95471686;
         Rs[7]<<0.10662873, 0.95030564, 0.29248849,
 -0.99415302, 0.10693514, 0.014989348,
 -0.017032834, -0.29237658, 0.9561516;
         Rs[8]<<0.11873601, 0.95415813, 0.2747435,
 -0.99274176, 0.1194065, 0.014346188,
 -0.019117631, -0.27445278, 0.96141052;
         Rs[9]<<0.11800467, 0.95603567, 0.26845971,
 -0.99247038, 0.12248512, 5.95483e-05,
 -0.032825392, -0.26644534, 0.96329099;

dVs[0]<<2.3447235, 0.031738445, -0.88403594;
dVs[1]<<2.3447235, 0.031738445, -0.88403594;
dVs[2]<<2.1667569, 0.053861078, -0.80424124;
dVs[3]<<2.1381314, 0.060161609, -0.77352905;
dVs[4]<<2.2968798, -0.0060790498, -0.83733064;
dVs[5]<<2.4576724, -0.0092556756, -0.95494056;
dVs[6]<<2.3961983, 0.012277888, -0.90623295;
dVs[7]<<2.280164, -0.023830272, -0.79988712;
dVs[8]<<2.1523359, 0.039374165, -0.78354788;
dVs[9]<<2.2297018, 0.020650063, -0.81908005;

dPs[0]<<0.30222359, 0.0033914039, -0.11917488;
dPs[1]<<0.30222359, 0.0033914039, -0.11917488;
dPs[2]<<0.27263474, 0.0033063588, -0.098992005;
dPs[3]<<0.26509905, 0.0059557888, -0.096052103;
dPs[4]<<0.28023139, 0.0010054347, -0.098338224;
dPs[5]<<0.30780178, -0.0029491552, -0.11862922;
dPs[6]<<0.29872048, 0.00070572889, -0.11245655;
dPs[7]<<0.29340258, -0.0033392629, -0.10331944;
dPs[8]<<0.26878262, 0.0018882495, -0.097184241;
dPs[9]<<0.27818239, 0.0014357395, -0.10081469;

dRs[0]<<0.99957997, -0.028765343, -0.003512128,
 0.02887797, 0.99886298, 0.037930958,
 0.0024170396, -0.038016446, 0.99927413;
 dRs[1]<<0.99957997, -0.028765343, -0.003512128,
 0.02887797, 0.99886298, 0.037930958,
 0.0024170396, -0.038016446, 0.99927413;
 dRs[2]<<0.99781948, -0.05799586, 0.031508539,
 0.056766864, 0.99764127, 0.038591918,
 -0.033672385, -0.036719132, 0.9987582;
 dRs[3]<<0.99849665, -0.051654968, -0.018335443,
 0.052934483, 0.99554813, 0.077986255,
 0.014225446, -0.07883957, 0.99678576;
 dRs[4]<<0.99874449, -0.046057943, -0.019698109,
 0.047906399, 0.99311697, 0.106881,
 0.014639816, -0.10769052, 0.99407667;
 dRs[5]<<0.9981823, -0.038267899, 0.046558287,
 0.035708878, 0.99786925, 0.054606669,
 -0.048548765, -0.052844848, 0.99742192;
 dRs[6]<<0.99969095, -0.019444486, 0.015487486,
 0.019203527, 0.99969453, 0.015557747,
 -0.015785255, -0.015255529, 0.99975902;
 dRs[7]<<0.99997574, 0.004440024, -0.0053650937,
 -0.0044567352, 0.99998522, -0.0031066926,
 0.005351224, 0.0031305298, 0.99998075;
 dRs[8]<<0.99949545, -0.031516559, 0.0039502843,
 0.03157948, 0.99935615, -0.01702822,
 -0.003411069, 0.017144388, 0.99984723;
 dRs[9]<<0.99958533, -0.022068495, 0.018498348,
 0.022179889, 0.99973696, -0.0058382517,
 -0.018364638, 0.0062461193, 0.99981183;

Cs[0]<<7.2250015e-09, 5.9243151e-18, -2.7183381e-16, -8.3455354e-11, 2.8973484e-09, -1.0313973e-10, -7.3885117e-12, 2.564608e-10, -8.5509213e-12, 0, 0, 0, 0, 0,0,
 5.595857e-18, 7.2250015e-09, 2.792843e-17, -2.8709546e-09, -3.8835571e-10, -8.0315274e-09, -2.5420085e-10, -3.3313435e-11, -6.8295136e-10, 0, 0, 0, 0, 0,0,
 -2.7197161e-16, 2.8822486e-17, 7.2249993e-09, -2.3812005e-10, 8.021007e-09, -3.0530234e-10, -2.0850268e-11, 6.8201478e-10, -2.5961545e-11, 0, 0, 0, 0, 0,0,
 -8.3455354e-11, -2.8709548e-09, -2.3812011e-10, 1.0016101e-06, -1.9688147e-10, 4.4307686e-09, 1.2516234e-07, -1.8379243e-11, 4.2884027e-10, 0, 0, 0, 0, 0,0,
 2.8973481e-09, -3.8835571e-10, 8.021007e-09, -1.9688152e-10, 1.0138386e-06, 7.1027031e-11, -1.7538217e-11, 1.2633694e-07, 6.338546e-12, 0, 0, 0, 0, 0,0,
 -1.0313971e-10, -8.0315266e-09, -3.0530228e-10, 4.4307678e-09, 7.102692e-11, 1.0122354e-06, 4.4360549e-10, 6.8960718e-12, 1.2617517e-07, 0, 0, 0, 0, 0,0,
 -7.3885108e-12, -2.5420085e-10, -2.0850269e-11, 1.2516232e-07, -1.7538216e-11, 4.4360551e-10, 2.0848727e-08, -1.6695664e-12, 4.5817947e-11, 0, 0, 0, 0, 0,0,
 2.5646088e-10, -3.3313435e-11, 6.8201494e-10, -1.8379244e-11, 1.2633696e-07, 6.8960926e-12, -1.6695667e-12, 2.0969058e-08, 6.2737538e-13, 0, 0, 0, 0, 0,0,
 -8.5509204e-12, -6.829512e-10, -2.5961537e-11, 4.2884021e-10, 6.3385248e-12, 1.2617515e-07, 4.5817954e-11, 6.2737256e-13, 2.0951623e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
Cs[1]<<7.2250015e-09, 5.9243151e-18, -2.7183381e-16, -8.3455354e-11, 2.8973484e-09, -1.0313973e-10, -7.3885117e-12, 2.564608e-10, -8.5509213e-12, 0, 0, 0, 0, 0,0,
 5.595857e-18, 7.2250015e-09, 2.792843e-17, -2.8709546e-09, -3.8835571e-10, -8.0315274e-09, -2.5420085e-10, -3.3313435e-11, -6.8295136e-10, 0, 0, 0, 0, 0,0,
 -2.7197161e-16, 2.8822486e-17, 7.2249993e-09, -2.3812005e-10, 8.021007e-09, -3.0530234e-10, -2.0850268e-11, 6.8201478e-10, -2.5961545e-11, 0, 0, 0, 0, 0,0,
 -8.3455354e-11, -2.8709548e-09, -2.3812011e-10, 1.0016101e-06, -1.9688147e-10, 4.4307686e-09, 1.2516234e-07, -1.8379243e-11, 4.2884027e-10, 0, 0, 0, 0, 0,0,
 2.8973481e-09, -3.8835571e-10, 8.021007e-09, -1.9688152e-10, 1.0138386e-06, 7.1027031e-11, -1.7538217e-11, 1.2633694e-07, 6.338546e-12, 0, 0, 0, 0, 0,0,
 -1.0313971e-10, -8.0315266e-09, -3.0530228e-10, 4.4307678e-09, 7.102692e-11, 1.0122354e-06, 4.4360549e-10, 6.8960718e-12, 1.2617517e-07, 0, 0, 0, 0, 0,0,
 -7.3885108e-12, -2.5420085e-10, -2.0850269e-11, 1.2516232e-07, -1.7538216e-11, 4.4360551e-10, 2.0848727e-08, -1.6695664e-12, 4.5817947e-11, 0, 0, 0, 0, 0,0,
 2.5646088e-10, -3.3313435e-11, 6.8201494e-10, -1.8379244e-11, 1.2633696e-07, 6.8960926e-12, -1.6695667e-12, 2.0969058e-08, 6.2737538e-13, 0, 0, 0, 0, 0,0,
 -8.5509204e-12, -6.829512e-10, -2.5961537e-11, 4.2884021e-10, 6.3385248e-12, 1.2617515e-07, 4.5817954e-11, 6.2737256e-13, 2.0951623e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[2]<<7.2249984e-09, -2.1890853e-16, -5.1527989e-16, -1.5439573e-10, 2.6288058e-09, -1.4344399e-10, -1.2900054e-11, 2.1228738e-10, -2.4383362e-11, 0, 0, 0, 0, 0,0,
 -2.2278984e-16, 7.225001e-09, 4.966755e-16, -2.8742075e-09, -4.4747167e-10, -7.6179481e-09, -2.3314942e-10, -3.6838092e-11, -6.3262745e-10, 0, 0, 0, 0, 0,0,
 -5.1544787e-16, 4.9712046e-16, 7.2249988e-09, -4.009362e-10, 7.7007671e-09, -2.8490871e-10, -2.062185e-11, 6.4003897e-10, -2.4080507e-11, 0, 0, 0, 0, 0,0,
 -1.5439573e-10, -2.8742078e-09, -4.0093617e-10, 1.0015575e-06, -3.5883485e-10, 4.0960271e-09, 1.2514083e-07, -3.1083448e-11, 3.8159104e-10, 0, 0, 0, 0, 0,0,
 2.6288056e-09, -4.4747167e-10, 7.7007654e-09, -3.5883499e-10, 1.0124091e-06, 1.3589103e-10, -1.6159579e-11, 1.2615627e-07, 6.1033414e-12, 0, 0, 0, 0, 0,0,
 -1.4344401e-10, -7.6179507e-09, -2.8490876e-10, 4.0960275e-09, 1.3589085e-10, 1.010878e-06, 3.7369316e-10, 1.1476016e-11, 1.2601649e-07, 0, 0, 0, 0, 0,0,
 -1.2900056e-11, -2.3314942e-10, -2.0621845e-11, 1.2514086e-07, -1.6159574e-11, 3.7369319e-10, 2.0844869e-08, -1.4661955e-12, 3.7128252e-11, 0, 0, 0, 0, 0,0,
 2.1228734e-10, -3.6838092e-11, 6.4003891e-10, -3.1083452e-11, 1.2615625e-07, 1.1476023e-11, -1.4661957e-12, 2.0946104e-08, 5.4024537e-13, 0, 0, 0, 0, 0,0,
 -2.4383363e-11, -6.3262767e-10, -2.4080508e-11, 3.8159106e-10, 6.1033392e-12, 1.2601647e-07, 3.7128255e-11, 5.4024612e-13, 2.0932518e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[3]<<7.2249997e-09, 4.6187118e-16, -8.6971935e-16, -1.4961084e-10, 2.8613769e-09, -1.4619396e-10, -1.2197337e-11, 2.3312166e-10, -1.1330299e-11, 0, 0, 0, 0, 0,0,
 4.6105639e-16, 7.2249997e-09, -1.7417221e-16, -2.7243219e-09, -7.4412609e-10, -7.611515e-09, -2.2188935e-10, -6.0712026e-11, -6.2126237e-10, 0, 0, 0, 0, 0,0,
 -8.6961056e-16, -1.735341e-16, 7.2250019e-09, -4.7237314e-10, 7.5570759e-09, -5.9993738e-10, -3.9121938e-11, 6.1679573e-10, -4.8976337e-11, 0, 0, 0, 0, 0,0,
 -1.4961088e-10, -2.7243223e-09, -4.7237303e-10, 1.0014207e-06, -3.6797421e-10, 3.9000563e-09, 1.2513036e-07, -3.3871069e-11, 3.5806216e-10, 0, 0, 0, 0, 0,0,
 2.8613771e-09, -7.4412598e-10, 7.5570759e-09, -3.6797365e-10, 1.0122101e-06, 1.3288635e-10, -3.4552378e-11, 1.261206e-07, 1.2482808e-11, 0, 0, 0, 0, 0,0,
 -1.4619396e-10, -7.6115132e-09, -5.9993749e-10, 3.9000554e-09, 1.3288641e-10, 1.0108142e-06, 3.579467e-10, 1.2233126e-11, 1.2599247e-07, 0, 0, 0, 0, 0,0,
 -1.2197339e-11, -2.2188935e-10, -3.9121931e-11, 1.2513037e-07, -3.455242e-11, 3.5794678e-10, 2.0844031e-08, -3.3800339e-12, 3.5097897e-11, 0, 0, 0, 0, 0,0,
 2.3312172e-10, -6.0712012e-11, 6.1679567e-10, -3.3871021e-11, 1.261206e-07, 1.2233111e-11, -3.3800265e-12, 2.0941066e-08, 1.2215803e-12, 0, 0, 0, 0, 0,0,
 -1.1330297e-11, -6.2126243e-10, -4.897633e-11, 3.5806216e-10, 1.2482817e-11, 1.2599247e-07, 3.5097897e-11, 1.2215813e-12, 2.0928507e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[4]<<7.225001e-09, 4.9158145e-16, -1.2162363e-15, -1.4971133e-10, 3.2652432e-09, -4.7152421e-10, -1.1303794e-11, 2.4629457e-10, -3.4803303e-11, 0, 0, 0, 0, 0,0,
 4.9051832e-16, 7.2250028e-09, -1.3593228e-16, -3.1333902e-09, -1.0420511e-09, -8.2696383e-09, -2.35438e-10, -8.3097793e-11, -6.6563038e-10, 0, 0, 0, 0, 0,0,
 -1.2160407e-15, -1.3685466e-16, 7.2250015e-09, -2.6427088e-10, 8.2189748e-09, -8.8892332e-10, -2.263132e-11, 6.6173333e-10, -7.1596604e-11, 0, 0, 0, 0, 0,0,
 -1.4971133e-10, -3.1333909e-09, -2.6427088e-10, 1.0017711e-06, 9.7134065e-11, 4.7537707e-09, 1.2514936e-07, 7.8406578e-12, 4.2579051e-10, 0, 0, 0, 0, 0,0,
 3.2652427e-09, -1.0420512e-09, 8.2189766e-09, 9.7134169e-11, 1.0145516e-06, -3.6561455e-11, 3.206343e-12, 1.2630312e-07, -1.2220776e-12, 0, 0, 0, 0, 0,0,
 -4.7152415e-10, -8.2696365e-09, -8.8892355e-10, 4.7537703e-09, -3.6561709e-11, 1.0127824e-06, 4.0452286e-10, -2.7691007e-12, 1.2615382e-07, 0, 0, 0, 0, 0,0,
 -1.1303795e-11, -2.35438e-10, -2.2631317e-11, 1.2514931e-07, 3.2063575e-12, 4.0452286e-10, 2.0844805e-08, 2.073528e-13, 3.8850697e-11, 0, 0, 0, 0, 0,0,
 2.4629448e-10, -8.3097793e-11, 6.6173345e-10, 7.8406457e-12, 1.2630314e-07, -2.7690918e-12, 2.0735003e-13, 2.0956044e-08, -7.4390581e-14, 0, 0, 0, 0, 0,0,
 -3.480331e-11, -6.6563022e-10, -7.1596597e-11, 4.257906e-10, -1.2220883e-12, 1.2615382e-07, 3.8850704e-11, -7.4389666e-14, 2.0942473e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[5]<<7.2250028e-09, -5.1731891e-16, -4.5774248e-16, -1.2056001e-10, 2.9743015e-09, -2.9109007e-10, -9.8723148e-12, 2.3928451e-10, -2.6978994e-11, 0, 0, 0, 0, 0,0,
 -5.1767843e-16, 7.2250015e-09, 4.3117603e-16, -3.3938108e-09, -5.8907546e-10, -8.6658627e-09, -2.7429323e-10, -4.8646927e-11, -7.1996908e-10, 0, 0, 0, 0, 0,0,
 -4.5851826e-16, 4.3264865e-16, 7.2250006e-09, -2.0475344e-10, 8.8196455e-09, -4.732989e-10, -1.3794453e-11, 7.324874e-10, -3.9458194e-11, 0, 0, 0, 0, 0,0,
 -1.2056002e-10, -3.3938115e-09, -2.0475348e-10, 1.0021361e-06, -1.7044796e-11, 5.4882654e-09, 1.2519406e-07, -7.0733203e-13, 5.1121524e-10, 0, 0, 0, 0, 0,0,
 2.9743015e-09, -5.8907529e-10, 8.8196446e-09, -1.7044586e-11, 1.0162446e-06, 6.8094913e-12, 3.6249068e-12, 1.2651266e-07, -1.3927001e-12, 0, 0, 0, 0, 0,0,
 -2.910901e-10, -8.6658662e-09, -4.7329884e-10, 5.4882663e-09, 6.8094259e-12, 1.0141088e-06, 5.0060328e-10, 2.7119762e-13, 1.2631853e-07, 0, 0, 0, 0, 0,0,
 -9.8723139e-12, -2.7429323e-10, -1.3794455e-11, 1.2519411e-07, 3.6248988e-12, 5.0060323e-10, 2.0850132e-08, 4.5397952e-13, 4.9791785e-11, 0, 0, 0, 0, 0,0,
 2.3928451e-10, -4.8646923e-11, 7.3248768e-10, -7.0732954e-13, 1.2651265e-07, 2.7118748e-13, 4.5398106e-13, 2.0981387e-08, -1.7294635e-13, 0, 0, 0, 0, 0,0,
 -2.6978997e-11, -7.199692e-10, -3.9458201e-11, 5.1121529e-10, -1.3926939e-12, 1.2631854e-07, 4.9791789e-11, -1.7294629e-13, 2.0962494e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[6]<<7.2250006e-09, -1.175144e-16, -8.7700203e-17, -6.1002599e-11, 3.0967873e-09, -9.5948111e-11, -5.0707815e-12, 2.561083e-10, -9.5681839e-12, 0, 0, 0, 0, 0,0,
 -1.1777585e-16, 7.2250006e-09, 1.4869424e-16, -3.2300462e-09, -1.9261492e-10, -8.5050926e-09, -2.6700819e-10, -1.5783066e-11, -6.9394063e-10, 0, 0, 0, 0, 0,0,
 -8.8820265e-17, 1.4907516e-16, 7.2249997e-09, -1.1769255e-10, 8.5543865e-09, -1.3129622e-10, -7.917049e-12, 6.9804967e-10, -1.0740089e-11, 0, 0, 0, 0, 0,0,
 -6.1002606e-11, -3.2300465e-09, -1.1769254e-10, 1.0019503e-06, -9.8145041e-11, 5.1137743e-09, 1.2518092e-07, -8.4264817e-12, 4.7016885e-10, 0, 0, 0, 0, 0,0,
 3.0967877e-09, -1.926149e-10, 8.5543848e-09, -9.8144944e-11, 1.0153657e-06, 3.7348739e-11, -5.7036758e-12, 1.2641311e-07, 2.1704045e-12, 0, 0, 0, 0, 0,0,
 -9.5948124e-11, -8.5050935e-09, -1.3129622e-10, 5.1137752e-09, 3.7348881e-11, 1.0134172e-06, 4.7418719e-10, 3.2408269e-12, 1.2623222e-07, 0, 0, 0, 0, 0,0,
 -5.0707797e-12, -2.6700817e-10, -7.9170498e-12, 1.251809e-07, -5.7036927e-12, 4.7418719e-10, 2.0849129e-08, -4.9876753e-13, 4.6516426e-11, 0, 0, 0, 0, 0,0,
 2.5610827e-10, -1.5783068e-11, 6.980494e-10, -8.4264609e-12, 1.2641314e-07, 3.2408234e-12, -4.9876509e-13, 2.0970079e-08, 1.9165099e-13, 0, 0, 0, 0, 0,0,
 -9.5681831e-12, -6.9394057e-10, -1.074009e-11, 4.701689e-10, 2.1704108e-12, 1.2623225e-07, 4.6516423e-11, 1.9165127e-13, 2.0952184e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[7]<<7.2250015e-09, 5.8059807e-18, -1.2779755e-18, 1.258693e-11, 2.7772953e-09, -3.9048768e-11, 1.0606967e-12, 2.349127e-10, -2.5652367e-12, 0, 0, 0, 0, 0,0,
 6.3519985e-18, 7.2250006e-09, 8.5343243e-18, -2.7351883e-09, 3.6657885e-11, -7.8303142e-09, -2.3136927e-10, 3.0915127e-12, -6.5937128e-10, 0, 0, 0, 0, 0,0,
 -1.3561357e-18, 8.4066928e-18, 7.2249993e-09, 8.2444024e-11, 7.8152738e-09, 2.4722452e-11, 6.2226968e-12, 6.5810229e-10, 2.0779476e-12, 0, 0, 0, 0, 0,0,
 1.2586928e-11, -2.7351881e-09, 8.2444031e-11, 1.0014141e-06, 1.0578664e-10, 4.0402259e-09, 1.2513512e-07, 9.9196215e-12, 3.8473774e-10, 0, 0, 0, 0, 0,0,
 2.7772955e-09, 3.6657892e-11, 7.8152755e-09, 1.0578663e-10, 1.0129645e-06, -3.7000774e-11, 9.0408046e-12, 1.2623435e-07, -3.1647625e-12, 0, 0, 0, 0, 0,0,
 -3.9048768e-11, -7.8303133e-09, 2.472245e-11, 4.0402259e-09, -3.7000757e-11, 1.0115522e-06, 3.8605677e-10, -3.4842402e-12, 1.2609929e-07, 0, 0, 0, 0, 0,0,
 1.0606966e-12, -2.3136923e-10, 6.2226977e-12, 1.251351e-07, 9.0408046e-12, 3.8605671e-10, 2.0845032e-08, 9.133125e-13, 3.926303e-11, 0, 0, 0, 0, 0,0,
 2.3491273e-10, 3.0915127e-12, 6.5810246e-10, 9.9196215e-12, 1.2623435e-07, -3.4842409e-12, 9.133125e-13, 2.0956776e-08, -3.2108994e-13, 0, 0, 0, 0, 0,0,
 -2.5652367e-12, -6.5937122e-10, 2.0779474e-12, 3.8473774e-10, -3.1647616e-12, 1.2609931e-07, 3.9263037e-11, -3.2108981e-13, 2.0942982e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
 Cs[8]<<7.2250019e-09, 3.2823162e-17, 1.3748998e-16, -8.7518458e-11, 2.768469e-09, -1.3917666e-11, -7.195345e-12, 2.2667712e-10, -9.7821083e-12, 0, 0, 0, 0, 0,0,
 3.4870897e-17, 7.2250024e-09, -4.9519746e-19, -2.7979907e-09, 4.2648291e-11, -7.6295459e-09, -2.2895849e-10, 3.5457065e-12, -6.2757627e-10, 0, 0, 0, 0, 0,0,
 1.3747136e-16, -1.7378031e-18, 7.2249988e-09, -1.794191e-10, 7.637178e-09, 1.3077639e-10, -6.1443797e-12, 6.2847216e-10, 1.0727681e-11, 0, 0, 0, 0, 0,0,
 -8.7518458e-11, -2.7979901e-09, -1.7941915e-10, 1.0014599e-06, -2.8532809e-10, 3.9656802e-09, 1.2513389e-07, -2.4630173e-11, 3.6677536e-10, 0, 0, 0, 0, 0,0,
 2.7684688e-09, 4.2648281e-11, 7.6371753e-09, -2.8532787e-10, 1.0122842e-06, 1.045629e-10, -1.416285e-11, 1.2613621e-07, 5.1893715e-12, 0, 0, 0, 0, 0,0,
 -1.3917691e-11, -7.6295494e-09, 1.3077639e-10, 3.9656807e-09, 1.0456292e-10, 1.0108398e-06, 3.6523071e-10, 8.983864e-12, 1.2600293e-07, 0, 0, 0, 0, 0,0,
 -7.1953446e-12, -2.289585e-10, -6.144381e-12, 1.2513391e-07, -1.4162862e-11, 3.6523065e-10, 2.0844357e-08, -1.27251e-12, 3.603759e-11, 0, 0, 0, 0, 0,0,
 2.2667708e-10, 3.5457065e-12, 6.2847205e-10, -2.4630159e-11, 1.2613621e-07, 8.983864e-12, -1.2725091e-12, 2.0943396e-08, 4.6419026e-13, 0, 0, 0, 0, 0,0,
 -9.7821118e-12, -6.2757638e-10, 1.0727682e-11, 3.6677544e-10, 5.1893689e-12, 1.2600296e-07, 3.6037603e-11, 4.6418988e-13, 2.0930308e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;
Cs[9]<<7.2250002e-09, 2.0828163e-17, 2.2627446e-17, -6.3372092e-11, 2.7986602e-09, -6.9253513e-11, -5.2538603e-12, 2.290501e-10, -9.3317836e-12, 0, 0, 0, 0, 0,0,
 2.0009562e-17, 7.2249997e-09, 9.9285662e-17, -2.9450176e-09, -1.5593726e-11, -7.9092874e-09, -2.410829e-10, -1.2518027e-12, -6.5142125e-10, 0, 0, 0, 0, 0,0,
 2.3490632e-17, 9.9377671e-17, 7.2250002e-09, -8.8997504e-11, 7.9620177e-09, 4.8139524e-11, -3.7114426e-12, 6.558174e-10, 3.8982086e-12, 0, 0, 0, 0, 0,0,
 -6.3372078e-11, -2.9450171e-09, -8.8997504e-11, 1.0016129e-06, -1.4033016e-10, 4.3325437e-09, 1.2514793e-07, -1.2365083e-11, 4.0057802e-10, 0, 0, 0, 0, 0,0,
 2.7986606e-09, -1.5593728e-11, 7.9620186e-09, -1.4033016e-10, 1.0132637e-06, 5.2197736e-11, -8.0917061e-12, 1.2622661e-07, 3.0037442e-12, 0, 0, 0, 0, 0,0,
 -6.925352e-11, -7.9092883e-09, 4.8139527e-11, 4.3325454e-09, 5.2197722e-11, 1.0116546e-06, 3.982796e-10, 4.5676128e-12, 1.2607883e-07, 0, 0, 0, 0, 0,0,
 -5.2538607e-12, -2.4108288e-10, -3.7114426e-12, 1.2514793e-07, -8.0917139e-12, 3.9827949e-10, 2.0845695e-08, -7.7901363e-13, 3.9237606e-11, 0, 0, 0, 0, 0,0,
 2.2905015e-10, -1.251803e-12, 6.5581734e-10, -1.2365075e-11, 1.2622662e-07, 4.5676128e-12, -7.7901271e-13, 2.095218e-08, 2.868481e-13, 0, 0, 0, 0, 0,0,
 -9.3317836e-12, -6.5142125e-10, 3.898209e-12, 4.0057802e-10, 3.0037435e-12, 1.2607883e-07, 3.9237613e-11, 2.8684797e-13, 2.0937728e-08, 0, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.402204e-11, 0, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06, 0,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06,0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2499994e-06;


 JRgs[0]<<-0.24995889, -0.0040205633, 0.00062931929,
 0.0040311376, -0.24988842, 0.005364095,
 -0.00052224688, -0.0053719962, -0.24992748 ;
 JRgs[1]<<-0.24995889, -0.0040205633, 0.00062931929,
 0.0040311376, -0.24988842, 0.005364095,
 -0.00052224688, -0.0053719962, -0.24992748 ;
 JRgs[2]<<-0.24978398, -0.0079874797, 0.0046310253,
 0.0080888392, -0.24978678, 0.0041843047,
 -0.0044520581, -0.0043694102, -0.24989793;
 JRgs[3]<<-0.24987733, -0.0055860723, -0.0036885843,
 0.0053773527, -0.24959148, 0.011309233,
 0.0040181638, -0.011206218, -0.24964333;
 JRgs[4]<<-0.24989319, -0.0061603263, -0.0010150505,
 0.0060570496, -0.24943921, 0.012949253,
 0.0014446991, -0.012901656, -0.24953316;
 JRgs[5]<<-0.24979708, -0.0041762567, 0.0079295011,
 0.0044098105, -0.24985309, 0.0056574955,
 -0.0077895205, -0.0058308002, -0.24975306;
 JRgs[6]<<-0.24998498, -0.001732833, -0.00040116059,
 0.0017357692, -0.24998346, 0.0013882983,
 0.00041721441, -0.0013916667, -0.24998884;
 JRgs[7]<<-0.24999802, 0.00052972208, -0.00050434918,
 -0.00052796362, -0.2499976, -0.00072379492,
 0.00050631387, 0.00072187773, -0.24999797;
 JRgs[8]<<-0.2499537, -0.0041594789, 0.00059852534,
 0.0041530817, -0.24994241, -0.0021157139,
 -0.00064547156, 0.0021032454, -0.24998695;
 JRgs[9]<<-0.24996237, -0.002611351, 0.0027322085,
 0.0026014475, -0.24998021, -0.00071113673,
 -0.0027420279, 0.00067371462, -0.24997953;


 JVgs[0]<<0.00081479421, 0.099567488, 0.0054488527,
 -0.100917, 0.0036120738, -0.2775977,
 -0.0022036682, 0.27809447, 0.0028345673 ;
 JVgs[1]<<0.00081479421, 0.099567488, 0.0054488527,
 -0.100917, 0.0036120738, -0.2775977,
 -0.0022036682, 0.27809447, 0.0028345673 ;
 JVgs[2]<<0.001453312, 0.099846758, 0.01150535,
 -0.097680002, 0.0053974288, -0.26445422,
 -0.0059427414, 0.26375151, 0.0037543585;
 JVgs[3]<<0.0020404269, 0.095154226, 0.010866556,
 -0.094552658, 0.0074096574, -0.26422137,
 -0.0032079529, 0.26410186, 0.0053275875;
 JVgs[4]<<0.0016444079, 0.10887985, 0.0016855302,
 -0.11150527, 0.012432125, -0.2868171,
 0.0067818086, 0.28784934, 0.010729348;
 JVgs[5]<<0.0016110964, 0.11763787, 0.0032375187,
 -0.1153432, 0.0081086438, -0.30124691,
 0.0034336939, 0.30035108, 0.0064957598;
 JVgs[6]<<0.00095649081, 0.11180659, 0.0031655722,
 -0.10733828, 0.003163283, -0.29597434,
 0.00030054164, 0.29433581, 0.0021627599;
 JVgs[7]<<-0.00010106902, 0.09465301, -0.0025328975,
 -0.095307805, -2.2342158e-05, -0.27070808,
 0.0022862346, 0.27094027, 6.7211986e-05;
 JVgs[8]<<0.00091114157, 0.096789725, 0.0073082494,
 -0.096571244, -0.00057790102, -0.26398295,
 -0.0053595863, 0.26398823, -0.0014993658;
 JVgs[9]<<0.00080045569, 0.10191263, 0.0034399023,
 -0.10077594, 0.0001745576, -0.2740863,
 -0.0014782979, 0.27369025, -0.00064929936;

 JVas[0]<<-0.24996902, 0.0031162035, 0.0013437622,
 -0.0031393669, -0.24992697, -0.0040089916,
 -0.0012732258, 0.0040270146, -0.24995029 ;
 JVas[1]<<-0.24996902, 0.0031162035, 0.0013437622,
 -0.0031393669, -0.24992697, -0.0040089916,
 -0.0012732258, 0.0040270146, -0.24995029 ;
 JVas[2]<<-0.24985421, 0.0062338803, -0.0034168749,
 -0.0061395792, -0.24982548, -0.005110254,
 0.003583116, 0.0049851108, -0.24990034;
 JVas[3]<<-0.24985684, 0.007391281, 0.00026429715,
 -0.00742803, -0.249661, -0.0082101421,
 6.2281164e-05, 0.0082309255, -0.24979296;
 JVas[4]<<-0.24989004, 0.0054750345, 0.003255937,
 -0.005681965, -0.24941342, -0.013591644,
 -0.0028378936, 0.01367758, -0.24947912;
 JVas[5]<<-0.24987894, 0.0050255624, -0.0038130756,
 -0.0048556295, -0.24979353, -0.0075730807,
 0.0040035094, 0.0074583059, -0.24979948;
 JVas[6]<<-0.24993582, 0.0030583444, -0.0042610099,
 -0.0030108541, -0.24996276, -0.0024702034,
 0.0042972541, 0.0024115082, -0.24994399;
 JVas[7]<<-0.249997, -0.00057287654, 0.00082039629,
 0.00057332165, -0.24999852, 4.7075813e-05,
 -0.00081986201, -4.9811813e-05, -0.2499982;
 JVas[8]<<-0.24996227, 0.0036494639, -0.00031274493,
 -0.0036530606, -0.24995026, 0.0021188133,
 0.00026990645, -0.0021251382, -0.24998745;
 JVas[9]<<-0.24996786, 0.0028637107, -0.0018311953,
 -0.0028719171, -0.24997695, 0.00079449493,
 0.0018192942, -0.00082425034, -0.24998757;

  JPgs[0]<< 5.3704181e-05, 0.0088163931, 0.00045023012,
 -0.0089179548, 0.00022765771, -0.023609726,
 -0.00024330647, 0.023648463, 0.00017729003;
 JPgs[1]<<5.3704181e-05, 0.0088163931, 0.00045023012,
 -0.0089179548, 0.00022765771, -0.023609726,
 -0.00024330647, 0.023648463, 0.00017729003;
 JPgs[2]<<8.8562498e-05, 0.0080935135, 0.00049575832,
 -0.0079604192, 0.00034285753, -0.021966593,
 -0.00015546716, 0.0219207, 0.00024872998;
 JPgs[3]<<0.00012882614, 0.0077579524, 0.00085994083,
 -0.0076901228, 0.00043761099, -0.021575799,
 -0.00038352553, 0.021558329, 0.00030434554;
 JPgs[4]<<8.9141904e-05, 0.0081918519, 0.00014242517,
 -0.0083665773, 0.00074239273, -0.023113552,
 0.00034246809, 0.02317719, 0.00065054215;
 JPgs[5]<<0.00010021385, 0.0095058177, 0.00011985402,
 -0.0093930569, 0.00052187347, -0.025003564,
 0.00030425552, 0.024960237, 0.00042175062;
 JPgs[6]<<6.3660002e-05, 0.0092422506, 0.00018517573,
 -0.0089288726, 0.00020660987, -0.024133656,
 4.063809e-05, 0.02401641, 0.00014071935;
 JPgs[7]<<-3.1337179e-06, 0.0080066361, -0.00018750556,
 -0.0080511346, 5.5821197e-06, -0.022799443,
 0.00018252048, 0.022815196, 8.0167874e-06;
 JPgs[8]<<5.4548764e-05, 0.0079231402, 0.00031435574,
 -0.0079116728, -3.5992765e-05, -0.021722065,
 -0.00019800536, 0.021720586, -9.0848967e-05;
 JPgs[9]<<5.1516592e-05, 0.0083431527, 0.00016205988,
 -0.0082825422, 1.0856892e-05, -0.022564959,
 -3.7381997e-05, 0.022543095, -4.1067357e-05;

  JPas[0]<<-0.031248037, 0.00025083456, 0.00014810846,
 -0.0002526888, -0.031245913, -0.00031468342,
 -0.00014415033, 0.00031615718, -0.031247057 ;
 JPas[1]<<-0.031248037, 0.00025083456, 0.00014810846,
 -0.0002526888, -0.031245913, -0.00031468342,
 -0.00014415033, 0.00031615718, -0.031247057 ;
 JPas[2]<<-0.031241983, 0.00048159817, -0.00026408851,
 -0.00047576398, -0.03123945, -0.00045061327,
 0.0002743646, 0.00044372652, -0.031243555;
 JPas[3]<<-0.031239811, 0.00066114007, -7.4712065e-05,
 -0.00066075311, -0.031229496, -0.00061638688,
 9.4698044e-05, 0.00061520992, -0.031238973;
 JPas[4]<<-0.031243095, 0.0004330576, 0.00030242561,
 -0.00044781168, -0.03121342, -0.0011292896,
 -0.00027709562, 0.0011351593, -0.031216763;
 JPas[5]<<-0.031243511, 0.0004270002, -0.00022822079,
 -0.00041839163, -0.031235857, -0.00066524581,
 0.00024112443, 0.00065955595, -0.031237736;
 JPas[6]<<-0.031244026, 0.00027713104, -0.00046203355,
 -0.00027283063, -0.031247234, -0.00022817992,
 0.00046474556, 0.00022298856, -0.031244561;
 JPas[7]<<-0.031249799, -3.1550622e-05, 7.4754287e-05,
 3.1540338e-05, -0.031249911, -8.3322384e-06,
 -7.474477e-05, 8.1823118e-06, -0.03124986;
 JPas[8]<<-0.031247808, 0.00028908599, -2.1758335e-05,
 -0.00028927551, -0.031247066, 0.00017289969,
 1.9189507e-05, -0.00017322256, -0.031249234;
 JPas[9]<<-0.03124808, 0.00024514552, -0.00012661511,
 -0.0002456175, -0.031248491, 7.037873e-05,
 0.00012581001, -7.2036746e-05, -0.031249356;



               Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
    if (!initialized)
    {
        Vector3d dirG = Vector3d::Zero();
       // bool isfirst=true;
        Vector3d velocity;
        bool firstframe=true;
        int i=1;
        for(std::vector<Frame::Ptr>::iterator iter_keyframe = Key_frames.begin()+1; iter_keyframe!=Key_frames.end(); iter_keyframe++) 
        {
            if(!(*iter_keyframe)->last_keyframe)
            {
                continue;
            }
            if(!(*iter_keyframe)->preintegration->isPreintegrated)
            {
                return false;
            }   
            // if(firstframe)
            // {
            //     (*iter_keyframe)->last_keyframe->time=times[0];
            //  (*iter_keyframe)->last_keyframe->pose=SE3d(Quaterniond(tcb*Rs[i]),tcb*Ps[0]);
            //   (*iter_keyframe)->last_keyframe->preintegration->dT=0.25;
            //   (*iter_keyframe)->last_keyframe->preintegration->dV=dVs[0];
            //   (*iter_keyframe)->last_keyframe->preintegration->dP=dPs[0];
            //   (*iter_keyframe)->last_keyframe->preintegration->dR=dRs[0];
            //     (*iter_keyframe)->last_keyframe->preintegration->C=Cs[0];
            //         (*iter_keyframe)->last_keyframe->preintegration->JRg=JRgs[0];
            //       (*iter_keyframe)->last_keyframe->preintegration->JVg=JVgs[0];
            //       (*iter_keyframe)->last_keyframe->preintegration->JVa=JVas[0];
            //       (*iter_keyframe)->last_keyframe->preintegration->JPg=JPgs[0];
            //       (*iter_keyframe)->last_keyframe->preintegration->JPa=JPas[0];
            //     firstframe=false;
            // }
            // (*iter_keyframe)->time=times[i];
            //  (*iter_keyframe)->pose=SE3d(Quaterniond(tcb*Rs[i]),tcb*Ps[i]);
            //   (*iter_keyframe)->preintegration->dT=0.25;
            //   (*iter_keyframe)->preintegration->dV=dVs[i];
            //   (*iter_keyframe)->preintegration->dP=dPs[i];
            //   (*iter_keyframe)->preintegration->dR=dRs[i];
            //     (*iter_keyframe)->preintegration->C=Cs[i];
            //       (*iter_keyframe)->preintegration->JRg=JRgs[i];
            //       (*iter_keyframe)->preintegration->JVg=JVgs[i];
            //       (*iter_keyframe)->preintegration->JVa=JVas[i];
            //       (*iter_keyframe)->preintegration->JPg=JPgs[i];
            //       (*iter_keyframe)->preintegration->JPa=JPas[i];

            i++;
            dirG -= (*iter_keyframe)->last_keyframe->GetImuRotation()*(*iter_keyframe)->preintegration->GetUpdatedDeltaVelocity();
            velocity= ((*iter_keyframe)->GetImuPosition() - (*(iter_keyframe))->last_keyframe->GetImuPosition())/((*iter_keyframe)->preintegration->dT);
            (*iter_keyframe)->SetVelocity(velocity);
            (*iter_keyframe)->last_keyframe->SetVelocity(velocity);
                
            LOG(INFO)<<"InitializeIMU  "<<(*iter_keyframe)->time-1.40364e+09/*+8.60223e+07*/<<"   Vwb1  "<<(tcb.inverse()*velocity).transpose()<<"  dt " <<(*iter_keyframe)->preintegration->dT;
           // LOG(INFO)<<"current  "<<(*iter_keyframe)->id<<"   last  "<< (*(iter_keyframe))->last_keyframe->id;
            // LOG(INFO)<<"current  "<<(tcb.inverse()*(*iter_keyframe)->GetImuPosition()).transpose()<<"   last  "<< (tcb.inverse()*(*(iter_keyframe))->last_keyframe->GetImuPosition()).transpose();
            //LOG(INFO)<<"  dP "<<(*iter_keyframe)->preintegration->dP.transpose()<<"  dV "<<(*iter_keyframe)->preintegration->dV.transpose();
        }
        dirG = dirG/dirG.norm();
  
        Vector3d  gI(0.0, 0.0, -1.0);//沿-z的归一化的重力数值
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v*ang/nv;
        Rwg = ExpSO3(vzg);
        //Rwg=Matrix3d::Identity();
        Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
        // LOG(INFO)<<"dirG "<<(tcb.inverse()*dirG).transpose();
       // LOG(INFO)<<"INITG "<<(tcb.inverse()*g).transpose();
    } 
    else
    {
        Rwg=Matrix3d::Identity();
        bg = Map::Instance().current_frame->GetGyroBias();
        ba =Map::Instance().current_frame->GetAccBias();
    }
    return true;
}


bool  Initializer::InitializeIMU(Frames keyframes)
{
    
    double priorA=1e5;
    double priorG=1e2;
    double minTime=20.0;  // 初始化需要的最小时间间隔
    // 按时间顺序收集初始化imu使用的KF
    std::list< Frame::Ptr > KeyFrames_list;
    Frames::reverse_iterator   iter;
        for(iter = keyframes.rbegin(); iter != keyframes.rend(); iter++)
    {
        KeyFrames_list.push_front(iter->second);
    }
    std::vector< Frame::Ptr > Key_frames(KeyFrames_list.begin(),KeyFrames_list.end());
    // imu计算初始时间
    FirstTs=Key_frames.front()->time;
    // LOG(INFO)<<FirstTs-Map::Instance().keyframes.begin()->second->time;
    // if(FirstTs-Map::Instance().keyframes.begin()->second->time<minTime)
    //     return false;

    const int N = Key_frames.size();  // 待处理的关键帧数目

    // 估计KF速度和重力方向
    if(!estimate_Vel_Rwg(Key_frames))
    {
        return false;
    }
 
    // KF速度和重力方向优化
    Scale=1.0;
                Vector3d g;
         g<< 0, 0, -G;
         g=Rwg*g;
                        Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
            //LOG(INFO)<<"INITG "<<(tcb.inverse()*g).transpose()<<"\n"<<Rwg;
    Bias bias_=InertialOptimization(keyframes,Rwg, Scale, bg, ba, priorG, priorA);

        Vector3d dirG;
         dirG<< 0, 0, -G;
            dirG=Rwg*dirG;
        // dirG<<-0.15915923848354485 ,  9.1792672332077689,   3.4306621858045498;
        // dirG=tcb*dirG;
        dirG = dirG/dirG.norm();
        Vector3d  gI(0.0, 0.0, -1.0);//沿-z的归一化的重力数值
        // 计算旋转轴
        Vector3d v = gI.cross(dirG);
        const double nv = v.norm();
        // 计算旋转角
        const double cosg = gI.dot(dirG);
        const double ang = acos(cosg);
        // 计算mRwg，与-Z旋转偏差
        Vector3d vzg = v*ang/nv;
        Rwg = ExpSO3(vzg);
                        Vector3d g2;
         g2<< 0, 0, -G;
         g2=Rwg*g2;
     LOG(INFO)<<"OPTG "<<(tcb.inverse()*g2).transpose()<<"\n"<<tcb.inverse()*Rwg;

    if (Scale<1e-1)
    {
        return false;
    }
    Map::Instance().ApplyScaledRotation(Rwg.inverse());
    // frontend_.lock()->UpdateFrameIMU(bias_);
    //更新关键帧中imu状态
    for(int i=0;i<N;i++)
    {
        Frame::Ptr pKF2 = Key_frames[i];
        pKF2->bImu = true;
    }

    // 进行完全惯性优化
    if(!initialized)
        Bias lastBias=FullInertialBA(keyframes,Rwg, priorG, priorA);
    // frontend_.lock()->UpdateFrameIMU(lastBias);
    frontend_.lock()->last_key_frame->bImu = true;
    bimu=true;
    initialized=true;
    reinit=false;
    return true;
}

} // namespace lvio_fusioni
