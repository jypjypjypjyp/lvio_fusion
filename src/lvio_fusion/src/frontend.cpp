#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Frontend::Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe)
    : num_features_(num_features), num_features_init_(init), num_features_tracking_bad_(tracking_bad),
      num_features_needed_for_keyframe_(need_for_keyframe)
{
}

bool Frontend::AddFrame(lvio_fusion::Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex);
      //NEWADD
    if(Imu::Num())
    {
        if(last_frame)
            frame->SetNewBias(last_frame->GetImuBias());
        frame->preintegration = imu::Preintegration::Create(frame->GetImuBias());
    }
    //NEWADDEND
    current_frame = frame;

    switch (status)
    {
    case FrontendStatus::BUILDING:
        InitMap();
        break;
    case FrontendStatus::INITIALIZING:
    case FrontendStatus::TRACKING_GOOD:
    case FrontendStatus::TRACKING_TRY:
        if (Track())
        {
            //NOTE: semantic map
            if (!current_frame->objects.empty())
            {
                current_frame->UpdateLabel();
            }
            break;
        }
        else
        {
            return false;
        }
    case FrontendStatus::LOST:
        //TODO
        Reset();
        InitMap();
        break;
    }
    last_frame = current_frame;
    last_frame_pose_cache_ = last_frame->pose;
    return true;
}

void Frontend::AddImu(double time, Vector3d acc, Vector3d gyr)
{
      //NEWADD
    imuPoint imuMeas(acc,gyr,time);
    imuData_buf.push_back(imuMeas);

    //NEWADDEND
}
void Frontend::PreintegrateIMU()
{
    if(last_frame)
    {
        if(backend_.lock()->initializer_->bimu)
        {
            current_frame->bImu=true;
        }
        std::vector<imuPoint> imuDatafromlastframe;
        while(true)
        {
            if(imuData_buf.empty())
            {
                usleep(500);
                continue;
            }
            imuPoint imudata = imuData_buf.front();
            if(imudata.t<last_frame->time-0.001)
            {
                last_imuData=imudata;
                imuData_buf.pop_front();
            }
            else if(imudata.t<current_frame->time-0.001)
            {
                imuDatafromlastframe.push_back(imudata);
                last_imuData=imudata;

                imuData_buf.pop_front();
            }
            else
            {
                imuDatafromlastframe.push_back(imudata);
                break;
            }
        }
        const int n = imuDatafromlastframe.size()-1;
        imu::Preintegration::Ptr   ImuPreintegratedFromLastFrame= imu::Preintegration::Create(last_frame->GetImuBias());
        for(int i=0; i<n; i++)
        {
            if( imuDatafromlastframe[i+1].t-imuDatafromlastframe[i].t>0.018){
                if(backend_.lock()->initializer_->initialized)
                    backend_.lock()->initializer_->reinit=true;   
                ImuPreintegratedFromLastKF->isPreintegrated=false;
                ImuPreintegratedFromLastFrame->isPreintegrated=false;
            }
            double tstep;
            double tab;
            Vector3d acc, angVel;
            if((i==0) && (i<(n-1)))
            {
                tab = imuDatafromlastframe[i+1].t-imuDatafromlastframe[i].t;
                double tini = imuDatafromlastframe[i].t- last_frame->time;
                acc = (imuDatafromlastframe[i].a+imuDatafromlastframe[i+1].a-
                        (imuDatafromlastframe[i+1].a-imuDatafromlastframe[i].a)*(tini/tab))*0.5f;
                angVel = (imuDatafromlastframe[i].w+imuDatafromlastframe[i+1].w-
                        (imuDatafromlastframe[i+1].w-imuDatafromlastframe[i].w)*(tini/tab))*0.5f;
                tstep = imuDatafromlastframe[i+1].t- last_frame->time;
            }
            else if(i<(n-1))
            {
                acc = (imuDatafromlastframe[i].a+imuDatafromlastframe[i+1].a)*0.5f;
                angVel = (imuDatafromlastframe[i].w+imuDatafromlastframe[i+1].w)*0.5f;
                tstep = imuDatafromlastframe[i+1].t-imuDatafromlastframe[i].t;
            }
            else if((i>0) && (i==(n-1)))
            {
                tab = imuDatafromlastframe[i+1].t-imuDatafromlastframe[i].t;
                double tend = imuDatafromlastframe[i+1].t-current_frame->time;
                acc = (imuDatafromlastframe[i].a+imuDatafromlastframe[i+1].a-
                        (imuDatafromlastframe[i+1].a-imuDatafromlastframe[i].a)*(tend/tab))*0.5f;
                angVel = (imuDatafromlastframe[i].w+imuDatafromlastframe[i+1].w-
                        (imuDatafromlastframe[i+1].w-imuDatafromlastframe[i].w)*(tend/tab))*0.5f;
                tstep = current_frame->time-imuDatafromlastframe[i].t;
            }
            else if((i==0) && (i==(n-1)))
            {
                acc = imuDatafromlastframe[i].a;
                angVel = imuDatafromlastframe[i].w;
                tstep = current_frame->time-last_frame->time;
            }
            if(tab==0)continue;
           // LOG(INFO)<<"ACC "<<acc.transpose()<<" ANG "<<angVel.transpose()<<" t "<<tstep;
            ImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
            ImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
        }
        //LOG(INFO)<<last_imuData.t-imuDatafromlastframe[0].t;
        if((n==0)){
            if(backend_.lock()->initializer_->initialized){
                backend_.lock()->initializer_->reinit=true;
            }
        }
        
        current_frame->preintegration=ImuPreintegratedFromLastKF;
        current_frame->preintegrationFrame=ImuPreintegratedFromLastFrame;
   

        // if(ImuPreintegratedFromLastKF->dT==0){
        //      backend_.lock()->initializer_->initialized=false;
        // }

    }   
}
bool Frontend::Track()
{
            //NEWADD
    if(Imu::Num())
    {
        PreintegrateIMU();
        //current_frame->preintegration->PreintegrateIMU(imuDatafromlastframe,last_frame->time, current_frame->time);
        if(backend_.lock()->initializer_->initialized)
        {
             PredictStateIMU();
        }
    }
   if(!Imu::Num()||!backend_.lock()->initializer_->initialized||(backend_.lock()->initializer_->initialized&&!current_frame->preintegrationFrame->isPreintegrated))
    {
        current_frame->pose = relative_i_j * last_frame_pose_cache_;
    }

    if(Imu::Num())
    {
        if(last_key_frame)
       {
            current_frame->last_keyframe = last_key_frame;
            current_frame->SetNewBias(last_key_frame->GetImuBias());
       }
    }

    //NEWADDEND
    //current_frame->pose = relative_i_j * last_frame_pose_cache_;
    // if(Imu::Num()&&backend_.lock()->initializer_->initialized) LocalBA();
    LocalBA();
    //if(!Imu::Num()||!backend_.lock()->initializer_->initialized)
    int num_inliers = TrackLastFrame(status == FrontendStatus::TRACKING_TRY ? last_key_frame : last_frame);

    if (status == FrontendStatus::INITIALIZING)
    {
        if (num_inliers <= num_features_tracking_bad_)
        {
            status = FrontendStatus::BUILDING;
        } 
    }
    else
    {
        if (num_inliers > num_features_tracking_bad_ &&
            (current_frame->pose.translation() - last_frame->pose.translation()).norm() < 5)
        {
            // tracking good
            status = FrontendStatus::TRACKING_GOOD;
        }
        else
        {
            // tracking bad, but give a chance
            status = FrontendStatus::TRACKING_TRY;
            LOG(INFO) << "Lost, try again!";
            return false;
        }
    }

    // Add every frame during initializing
    if (num_inliers < num_features_needed_for_keyframe_)
    {
        CreateKeyframe();
    }
    else if (status == FrontendStatus::INITIALIZING)
    {
        CreateKeyframe(false);
    }
    relative_i_j = current_frame->pose * last_frame_pose_cache_.inverse();
    return true;
}

inline double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

inline void optical_flow(cv::Mat &prevImg, cv::Mat &nextImg,
                                 std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                                 std::vector<uchar> &status)
{
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(
        prevImg, nextImg, prevPts, nextPts, status, err, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = prevPts;
    cv::calcOpticalFlowPyrLK(
        nextImg, prevImg, nextPts, reverse_pts, reverse_status, err, cv::Size(3, 3), 1,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    for (size_t i = 0; i < status.size(); i++)
    {
        // clang-format off
        if (status[i] && reverse_status[i] && distance(prevPts[i], reverse_pts[i]) <= 0.5
        && nextPts[i].x >= 0 && nextPts[i].x < prevImg.cols
        && nextPts[i].y >= 0 && nextPts[i].y < prevImg.rows)
        // clang-format on
        {
            status[i] = 1;
        }
        else
            status[i] = 0;
    }
}

int Frontend::TrackLastFrame(Frame::Ptr last_frame)
{
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    std::vector<uchar> status;
    // use LK flow to estimate points in the last image
    for (auto &pair_feature : last_frame->features_left)
    {
        // use project point
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        auto px = Camera::Get()->World2Pixel(position_cache_[camera_point->id], current_frame->pose);
        landmarks.push_back(camera_point);
        kps_last.push_back(feature->keypoint);
        kps_current.push_back(cv::Point2f(px[0], px[1]));
    }
    optical_flow(last_frame->image_left, current_frame->image_left, kps_last, kps_current, status);

    // mismatch points, try again by ORB mathcer
    // int a = mather_.Search(current_frame, last_frame, kps_current, kps_last, status);
    // LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << a;
    // Solve PnP
    std::unordered_map<int, int> map;
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            map[points_2d.size()] = i;
            points_2d.push_back(kps_current[i]);
            Vector3d p = position_cache_[landmarks[i]->id];
            points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
        }
    }

    cv::Mat rvec, tvec, inliers, cv_R;
    int num_good_pts = 0;
    if (points_2d.size() > num_features_tracking_bad_ && cv::solvePnPRansac(points_3d, points_2d, Camera::Get()->K, Camera::Get()->D, rvec, tvec, false, 100, 8.0F, 0.98, inliers, cv::SOLVEPNP_EPNP))
    {
        //DEBUG
        cv::Mat img_track = current_frame->image_left;
        cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
        for (int r = 0; r < inliers.rows; r++)
        {
            int i = map[inliers.at<int>(r)];
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            cv::circle(img_track, kps_current[i], 2, cv::Scalar(255, 0, 0), cv::FILLED);
            auto feature = visual::Feature::Create(current_frame, kps_current[i], landmarks[i]);
            current_frame->AddFeature(feature);
            num_good_pts++;
        }
        cv::imshow("tracking", img_track);
        cv::waitKey(1);

        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->pose = (Camera::Get()->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::InitMap()
{
    int num_new_features = DetectNewFeatures();
    if (num_new_features < num_features_init_)
    {
        return false;
    }
    if (Imu::Num())
    {
        status = FrontendStatus::INITIALIZING;
    }
    else
    {
        status = FrontendStatus::TRACKING_GOOD;
    }

    // the first frame is a keyframe
    Map::Instance().InsertKeyFrame(current_frame);
    LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend and loop because we have a new keyframe
    backend_.lock()->UpdateMap();
    return true;
}

int Frontend::DetectNewFeatures()
{
    int num_times = 0;
    int num_triangulated_pts = 0;
    int num_good_pts = 0;
    while (num_times++ < 2 && current_frame->features_left.size() < 0.8 * num_features_)
    {
        cv::Mat mask(current_frame->image_left.size(), CV_8UC1, 255);
        for (auto &pair_feature : current_frame->features_left)
        {
            auto feature = pair_feature.second;
            cv::circle(mask, feature->keypoint, 20, 0, cv::FILLED);
        }

        std::vector<cv::Point2f> kps_left, kps_right; // must be point2f
        cv::goodFeaturesToTrack(current_frame->image_left, kps_left, num_features_ - current_frame->features_left.size(), 0.01, 20, mask);

        // use LK flow to estimate points in the right image
        kps_right = kps_left;
        std::vector<uchar> status;
        optical_flow(current_frame->image_left, current_frame->image_right, kps_left, kps_right, status);

        // triangulate new points
        for (size_t i = 0; i < kps_left.size(); ++i)
        {
            if (status[i])
            {
                num_good_pts++;
                // triangulation
                Vector2d kp_left = cv2eigen(kps_left[i]);
                Vector2d kp_right = cv2eigen(kps_right[i]);
                Vector3d pb = Vector3d::Zero();
                triangulate(Camera::Get()->extrinsic.inverse(), Camera::Get(1)->extrinsic.inverse(),
                            Camera::Get()->Pixel2Sensor(kp_left), Camera::Get(1)->Pixel2Sensor(kp_right), pb);
                if ((Camera::Get()->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (Camera::Get(1)->Robot2Pixel(pb) - kp_right).norm() < 0.5)
                {
                    auto new_landmark = visual::Landmark::Create(pb);
                    auto new_left_feature = visual::Feature::Create(current_frame, eigen2cv(kp_left), new_landmark);
                    auto new_right_feature = visual::Feature::Create(current_frame, eigen2cv(kp_right), new_landmark);
                    new_right_feature->is_on_left_image = false;
                    new_landmark->AddObservation(new_left_feature);
                    new_landmark->AddObservation(new_right_feature);
                    current_frame->AddFeature(new_left_feature);
                    current_frame->AddFeature(new_right_feature);
                    Map::Instance().InsertLandmark(new_landmark);
                    position_cache_[new_landmark->id] = new_landmark->ToWorld();
                    num_triangulated_pts++;
                }
            }
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    LOG(INFO) << "new landmarks: " << num_triangulated_pts;
    return num_triangulated_pts;
}

void Frontend::CreateKeyframe(bool need_new_features)
{
        //NEWADD
    if(Imu::Num())
    { 
        ImuPreintegratedFromLastKF=imu::Preintegration::Create(current_frame->GetImuBias());
    }
    //NEWADDEND
    // first, add new observations of old points
    for (auto &pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        landmark->AddObservation(feature);
    }
    // detect new features, track in right image and triangulate map points
    if (need_new_features)
    {
        DetectNewFeatures();
    }
    // insert!
    Map::Instance().InsertKeyFrame(current_frame);
    last_key_frame = current_frame;
    LOG(INFO) << "Add a keyframe " << current_frame->id;
    // update backend because we have a new keyframe
    backend_.lock()->UpdateMap();
}

// TODO
bool Frontend::Reset()
{
    backend_.lock()->Pause();
    Map::Instance().Reset();
    backend_.lock()->Continue();
    status = FrontendStatus::BUILDING;
     //NEWADD
    current_frame=Frame::Create();
    last_frame=Frame::Create();
   last_key_frame=static_cast<Frame::Ptr>(NULL);
    //NEWADDEND
    LOG(INFO) << "Reset Succeed";
    return true;
}

void Frontend::UpdateCache()
{
    position_cache_.clear();
    for (auto &pair_feature : last_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        position_cache_[camera_point->id] = camera_point->ToWorld();
    }
    last_frame_pose_cache_ = last_frame->pose;
}
//NEWADD
void Frontend::UpdateFrameIMU( const Bias &bias_)
{
    last_frame->SetNewBias(bias_);
    current_frame->SetNewBias(bias_);

    Vector3d Gz ;
    Gz << 0, 0, -Imu::Get()->G;
   // Gz =backend_.lock()->initializer_->Rwg*Gz;

    Vector3d twb1;
    Matrix3d Rwb1;
   Vector3d Vwb1;
    double t12;  // 时间间隔
    Vector3d twb2;
    Matrix3d Rwb2;
   Vector3d Vwb2;
    // Step 2:更新Frame的pose velocity
    // Step 2.1:更新lastFrame的pose velocity
    if(last_frame->last_keyframe&&last_frame->preintegration->isPreintegrated){
        if(fabs(last_frame->time-last_key_frame->time)>0.001)
        {
            twb1= last_frame->last_keyframe->GetImuPosition();
            Rwb1 = last_frame->last_keyframe->GetImuRotation();
            Vwb1 = last_frame->last_keyframe->GetVelocity();
            t12 = last_frame->preintegration->dT;
            Rwb2=Rwb1*last_frame->preintegration->GetUpdatedDeltaRotation();
            twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*last_frame->preintegration->GetUpdatedDeltaPosition();
            Vwb2=Vwb1 + Gz*t12 + Rwb1*last_frame->preintegration->GetUpdatedDeltaVelocity();
           // last_frame->SetPose(Rwb2,twb2);
            last_frame->SetVelocity(Vwb2);
        }
    }
    // Step 2.2:更新currentFrame的pose velocity
    if (fabs(current_frame->time-last_key_frame->time)>0.001&&current_frame->preintegration->isPreintegrated&&current_frame->last_keyframe)
    {
         twb1= current_frame->last_keyframe->GetImuPosition();
        Rwb1 = current_frame->last_keyframe->GetImuRotation();
        Vwb1 = current_frame->last_keyframe->GetVelocity();
        t12 = current_frame->preintegration->dT;
        Rwb2=Rwb1*current_frame->preintegration->GetUpdatedDeltaRotation();
        twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_frame->preintegration->GetUpdatedDeltaPosition();
        Vwb2=Vwb1 + Gz*t12 + Rwb1*current_frame->preintegration->GetUpdatedDeltaVelocity();
       // current_frame->SetPose(Rwb2,twb2);
        current_frame->SetVelocity(Vwb2);
    }
}

void Frontend::PredictStateIMU()
{
            Matrix3d tcb;
            // tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            // 0.999557249008, 0.0149672133247, 0.025715529948,
            // -0.0257744366974, 0.00375618835797, 0.999660727178;
             tcb<<1,0,0,
             0,1,0,
             0,0,1;
    if(Map::Instance().mapUpdated&&last_key_frame)
    {
        Vector3d Gz ;
        Gz << 0, 0, -9.81007;
        //Gz =backend_.lock()->initializer_->Rwg*Gz;
        double t12=current_frame->preintegration->dT;
         Vector3d twb1=last_key_frame->GetImuPosition();
        Matrix3d Rwb1=last_key_frame->GetImuRotation();
        Vector3d Vwb1=last_key_frame->Vw;

        Matrix3d Rwb2=NormalizeRotation(Rwb1*current_frame->preintegration->GetDeltaRotation(last_key_frame->GetImuBias()));
        Vector3d twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_frame->preintegration->GetDeltaPosition(last_key_frame->GetImuBias());
        Vector3d Vwb2=Vwb1+t12*Gz+Rwb1*current_frame->preintegration->GetDeltaVelocity(last_key_frame->GetImuBias());
            //LOG(INFO)<<"T2"<<twb2.transpose()<<"V2"<<Vwb2.transpose();
        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2,twb2);
        current_frame->SetNewBias(last_key_frame->GetImuBias());
        Map::Instance().mapUpdated=false;//NEWADD
        //          LOG(INFO)<<"PredictStateIMU  "<<current_frame->time-1.40364e+09<<"  T12  "<<t12;
        // LOG(INFO)<<" Rwb2\n"<<tcb.inverse()*Rwb2;
    }
    else if(! Map::Instance().mapUpdated)
    {
           
        Vector3d Gz ;
        Gz << 0, 0, -9.81007;
          
       // Gz =backend_.lock()->initializer_->Rwg*Gz;
        double t12=current_frame->preintegrationFrame->dT;
         Vector3d twb1=last_frame->GetImuPosition();
        Matrix3d Rwb1=last_frame->GetImuRotation();
        Vector3d Vwb1=last_frame->Vw;
        // Bias bias_( -0.041236103, 0.0031201241 ,0.013771472,-0.0034145617, 0.02168173,0.078614868);
        // last_frame->SetNewBias(bias_);
        Matrix3d Rwb2=NormalizeRotation(Rwb1*current_frame->preintegrationFrame->GetDeltaRotation(last_frame->GetImuBias()));
        Vector3d twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_frame->preintegrationFrame->GetDeltaPosition(last_frame->GetImuBias());
        Vector3d Vwb2=Vwb1+t12*Gz+Rwb1*current_frame->preintegrationFrame->GetDeltaVelocity(last_frame->GetImuBias());
         
        //  LOG(INFO)<<"PredictStateIMU  "<<current_frame->time-1.40364e+09<<"  T12  "<<t12;
        // // LOG(INFO)<<" Rwb2\n"<<tcb.inverse()*Rwb2;
        // LOG(INFO)<<"  Vwb1  "<<Vwb1.transpose()*tcb;
        // LOG(INFO)<<" Vwb2  "<<Vwb2.transpose()*tcb;
        // LOG(INFO)<<"   GRdV   "<<(t12*Gz+Rwb1*current_frame->preintegrationFrame->GetDeltaVelocity(last_frame->GetImuBias())).transpose()*tcb;
        // LOG(INFO)<<"   RdV    "<<((tcb.inverse()*Rwb1)*current_frame->preintegrationFrame->GetDeltaVelocity(last_frame->GetImuBias())).transpose();
        // LOG(INFO)<<"   dV      "<<(current_frame->preintegrationFrame->GetDeltaVelocity(last_frame->GetImuBias())).transpose();
        // // //  LOG(INFO)<<"  dR:\n"<<current_frame->preintegrationFrame->GetDeltaRotation(last_frame->GetImuBias()); 
        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2,twb2);
        current_frame->SetNewBias(last_frame->GetImuBias());
    }

}
  bool showIMUErrorfornt(const double*  parameters0, const double*  parameters1, const double*  parameters2, const double*  parameters3, const double*  parameters4, const double*  parameters5, imu::Preintegration::Ptr mpInt,double time)  
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
      //  Matrix<double, 9,9> sqrt_info =LLT<Matrix<double, 9, 9>>( mpInt->C.block<9,9>(0,0).inverse()).matrixL().transpose();
        sqrt_info/=InfoScale;
        // LOG(INFO)<<"InertialError sqrt_info "<<sqrt_info;
        //assert(!isnan(residual[0])&&!isnan(residual[1])&&!isnan(residual[2])&&!isnan(residual[3])&&!isnan(residual[4])&&!isnan(residual[5])&&!isnan(residual[6])&&!isnan(residual[7])&&!isnan(residual[8]));
        residual = sqrt_info* residual;
     
        LOG(INFO)<<time<<" IMUError:  r "<<residual.transpose()<<"  "<<mpInt->dT;
         Matrix3d tcb;
            tcb<<0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008, 0.0149672133247, 0.025715529948,
            -0.0257744366974, 0.00375618835797, 0.999660727178;
        // LOG(INFO)<<"                Qi "<<Qi.toRotationMatrix().eulerAngles(0,1,2).transpose()<<" Qj "<<Qj.toRotationMatrix().eulerAngles(0,1,2).transpose()<<"dQ"<<dR.eulerAngles(0,1,2).transpose();
        // LOG(INFO)<<"                Pi "<<Pi.transpose()<<" Pj "<<Pj.transpose()<<"dP"<<dP.transpose();
        LOG(INFO)<<"                Vi "<<Vi.transpose()*tcb<<" Vj "<<Vj.transpose()*tcb<<"dV"<<dV.transpose();
        // LOG(INFO)<<"             Bai "<< accBias.transpose()<<"  Bgi "<<  gyroBias.transpose();
         return true;
    }
 void Frontend::LocalBA(){
     adapt::Problem problem;  
      ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
          ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
      auto para_kf=current_frame->pose.data();
      problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
      for (auto pair_feature : current_frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame().lock();
            ceres::CostFunction *cost_function;
            cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint),position_cache_[landmark->id],  Camera::Get(), current_frame->weights.visual);
            problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para_kf);
        }

       if(current_frame->last_keyframe&&backend_.lock()->initializer_->initialized&&current_frame->preintegration->isPreintegrated)
        {
            auto para_kf_last=current_frame->last_keyframe->pose.data();
            auto para_v=current_frame->Vw.data();
            auto para_v_last=current_frame->last_keyframe->Vw.data();
            auto para_accBias=current_frame->ImuBias.linearized_ba.data();
            auto para_gyroBias=current_frame->ImuBias.linearized_bg.data();
            auto para_accBias_last=current_frame->last_keyframe->ImuBias.linearized_ba.data();
            auto para_gyroBias_last=current_frame->last_keyframe->ImuBias.linearized_bg.data();
            problem.AddParameterBlock(para_kf_last, SE3d::num_parameters, local_parameterization);
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_v_last, 3);
            problem.AddParameterBlock(para_accBias, 3);
            problem.AddParameterBlock(para_gyroBias, 3);
            problem.AddParameterBlock(para_accBias_last, 3);
            problem.AddParameterBlock(para_gyroBias_last, 3);
            problem.SetParameterBlockConstant(para_kf_last);

            problem.SetParameterBlockConstant(para_accBias_last);
            problem.SetParameterBlockConstant(para_gyroBias_last);

             ceres::CostFunction *cost_function =InertialError3::Create(current_frame->preintegration,backend_.lock()->initializer_->Rwg);
            problem.AddResidualBlock(ProblemType::IMUError,cost_function, NULL, para_kf_last, para_v_last,  para_gyroBias_last,para_accBias_last, para_kf, para_v);
            ceres::CostFunction *cost_function_g = GyroRWError3::Create(current_frame->preintegration->C.block<3,3>(9,9).inverse());
            problem.AddResidualBlock(ProblemType::IMUError,cost_function_g, NULL, para_gyroBias_last,para_gyroBias);
            ceres::CostFunction *cost_function_a = AccRWError3::Create(current_frame->preintegration->C.block<3,3>(12,12).inverse());
            problem.AddResidualBlock(ProblemType::IMUError,cost_function_a, NULL, para_accBias_last,para_accBias);
            //  showIMUErrorfornt(para_kf_last, para_v_last,  para_gyroBias_last,para_accBias_last, para_kf, para_v,current_frame->preintegration,current_frame->time-1.40364e+09+8.60223e+07);
        }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.1;
   // options.max_num_iterations = 4;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

     if(current_frame->last_keyframe&&backend_.lock()->initializer_->initialized)
        {
            auto para_kf_last=current_frame->last_keyframe->pose.data();
            auto para_v=current_frame->Vw.data();
            auto para_v_last=current_frame->last_keyframe->Vw.data();
            auto para_accBias=current_frame->ImuBias.linearized_ba.data();
            auto para_gyroBias=current_frame->ImuBias.linearized_bg.data();
            auto para_accBias_last=current_frame->last_keyframe->ImuBias.linearized_ba.data();
            auto para_gyroBias_last=current_frame->last_keyframe->ImuBias.linearized_bg.data();
            //  showIMUErrorfornt(para_kf_last, para_v_last,  para_gyroBias_last,para_accBias_last, para_kf, para_v,current_frame->preintegration,current_frame->time-1.40364e+09);

        }
 }
//NEWADDEND
} // namespace lvio_fusion