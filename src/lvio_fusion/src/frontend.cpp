#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Frontend::Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe)
    : num_features_(num_features), num_features_init_(init), num_features_tracking_(tracking),
      num_features_tracking_bad_(tracking_bad), num_features_needed_for_keyframe_(need_for_keyframe)
{
}

bool Frontend::AddFrame(lvio_fusion::Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex);
    //NEWADD
    frame->calib_= ImuCalib_;
    //frame->pose = relative_pose * last_frame_pose_cache_;
    frame->preintegration = imu::Preintegration::Create(frame->GetImuBias(), ImuCalib_,NULL);
    //NEWADDEND
    current_frame = frame;

    switch (status)
    {
    case FrontendStatus::BUILDING:
        InitMap();
        break;
    case FrontendStatus::INITIALIZING:
    case FrontendStatus::TRACKING_GOOD:
    case FrontendStatus::TRACKING_BAD:
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
      static bool first = true;
     if (current_frame)
    {
        // if (!current_frame->preintegration)
        // {
        //     current_frame->preintegration = imu::Preintegration::Create(current_frame->GetImuBias(), ImuCalib_,NULL);
        // }
        current_frame->preintegration->Appendimu(imuMeas);
        if (current_key_frame && current_key_frame->preintegration && current_key_frame != current_frame)
        {
            current_key_frame->preintegration->Appendimu(imuMeas);
        }

    }
    //NEWADDEND
}

bool Frontend::Track()
{    
    current_frame->pose = relative_i_j * last_frame_pose_cache_;
    //NEWADD
     //LOG(INFO)<<" FRAME ID: "<<current_frame->id<<"  pose: "<< current_frame->pose.translation()[0]<<" "<<current_frame->pose.translation()[1]<<" "<<current_frame->pose.translation()[2] ;
    //LOG(INFO)<<" relative_pose"<<relative_pose.translation()[0]<<" "<<relative_pose.translation()[1]<<" "<<relative_pose.translation()[2];
    //LOG(INFO)<<" last_frame_pose_cache_"<<last_frame_pose_cache_.translation()[0]<<" "<<last_frame_pose_cache_.translation()[1]<<" "<<last_frame_pose_cache_.translation()[2];
     //如果有imu  预积分上一帧到当前帧的imu 
    if(imu_&&last_frame&&last_frame->preintegration){
        // if (!current_frame->preintegration)
        // {
        //     current_frame->preintegration = imu::Preintegration::Create(current_frame->GetImuBias(), ImuCalib_,imu_);
        // }
        current_frame->preintegration->PreintegrateIMU( last_frame->preintegration->imuData_buf,last_frame->time, current_frame->time);//TODO 这个结果应该存在current_frame
       if(last_key_frame)
       {  
            current_frame->mpLastKeyFrame = last_key_frame;
            current_frame->SetNewBias(last_key_frame->GetImuBias());
       }
    }
//NEWADDEND
    TrackLastFrame();
    InitFramePoseByPnP();
    int inliers = current_frame->features_left.size();

    static int num_tries = 0;
    if (status == FrontendStatus::INITIALIZING)
    {
        if (inliers <= num_features_tracking_bad_)
        {
            status = FrontendStatus::BUILDING;
        }
    }
    else
    {
        if (inliers > num_features_tracking_)
        {
            // tracking good
            status = FrontendStatus::TRACKING_GOOD;
            num_tries = 0;
        }
        else if (inliers > num_features_tracking_bad_)
        {
            // tracking bad
            status = FrontendStatus::TRACKING_BAD;
            num_tries = 0;
        }
        else
        {
            // lost, but give a chance
            num_tries++;
            status = num_tries >= 4 ? FrontendStatus::LOST : FrontendStatus::TRACKING_TRY;
            num_tries %= 4;
            return false;
        }
    }

    // Add every frame during initializing
    if (inliers < num_features_needed_for_keyframe_)
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

void Frontend::CreateKeyframe(bool need_new_features)
{
        // //NEWADD
    if(imu_){    //如果有imu  预积分上一关键帧到当前帧的imu 
        // if(backend_.lock()->initializer_ ->bInitializing)//初始化时不能创建关键帧
        // {
        //     return;
        // }
        last_key_frame=current_key_frame;
        }
        //NEWADDEND
    // first, add new observations of old points
    for (auto pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto mp = feature->landmark.lock();
        mp->AddObservation(feature);
    }
    // detect new features, track in right image and triangulate map points
    if (need_new_features)
    {
        DetectNewFeatures();
    }
    // insert!
    map_->InsertKeyFrame(current_frame);
    current_key_frame = current_frame;

//NEWADD
    if(imu_&&last_key_frame&&last_key_frame->preintegration)
    {
        if(backend_.lock()->initializer_->bimu)
        {
            current_key_frame->bImu=true;
        }  
        //  if (!current_key_frame->preintegration)
        // {
        //     current_key_frame->preintegration = imu::Preintegration::Create(current_key_frame->GetImuBias(), ImuCalib_,imu_);
        // }
        current_key_frame->preintegration->PreintegrateIMU(last_key_frame->preintegration->imuData_buf,last_key_frame->time, current_key_frame->time);
    }
    reference_key_frame=current_key_frame;
//NEWADDEND

    //LOG(INFO) << "Add a keyframe " << current_frame->id;
    // update backend because we have a new keyframe
    backend_.lock()->UpdateMap();
}

bool Frontend::InitFramePoseByPnP()
{
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        points_2d.push_back(feature->keypoint);
        Vector3d p = position_cache_[camera_point->id];
        points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
    }

    cv::Mat K;
    cv::eigen2cv(camera_left_->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (cv::solvePnPRansac(points_3d, points_2d, K, D, rvec, tvec, false, 100, 8.0F, 0.98, cv::noArray(), cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->pose = (camera_left_->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();
        return true;
    }
    return false;
}

int Frontend::TrackLastFrame()
{
    // use LK flow to estimate points in the last image
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    for (auto pair_feature : last_frame->features_left)
    {
        // use project point
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        auto px = camera_left_->World2Pixel(position_cache_[camera_point->id], current_frame->pose);
        landmarks.push_back(camera_point);
        kps_last.push_back(feature->keypoint);
        kps_current.push_back(cv::Point2f(px[0], px[1]));
    }

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame->image_left, current_frame->image_left, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // NOTE: don‘t use, accuracy is very low
    // cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, 3.0, 0.9, status);

    int num_good_pts = 0;
    //DEBUG
    cv::Mat img_track = current_frame->image_left;
    cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            auto feature = visual::Feature::Create(current_frame, kps_current[i], landmarks[i]);
            current_frame->AddFeature(feature);
            num_good_pts++;
        }
    }
    cv::imshow("tracking", img_track);
    cv::waitKey(1);
    //LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::InitMap()
{
    int num_new_features = DetectNewFeatures();
    if (num_new_features < num_features_init_)
    {
        return false;
    }
    if (imu_)
    {
        status = FrontendStatus::INITIALIZING;
    }
    else
    {
        status = FrontendStatus::TRACKING_GOOD;
    }

    // the first frame is a keyframe
    map_->InsertKeyFrame(current_frame);
    //LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend and loop because we have a new keyframe
    backend_.lock()->UpdateMap();
    return true;
}

int Frontend::DetectNewFeatures()
{
    cv::Mat mask(current_frame->image_left.size(), CV_8UC1, 255);
    for (auto pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        cv::rectangle(mask, feature->keypoint - cv::Point2f(10, 10), feature->keypoint + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::Point2f> kps_left, kps_right; // must be point2f
    cv::goodFeaturesToTrack(current_frame->image_left, kps_left, num_features_ - current_frame->features_left.size(), 0.01, 30, mask);

    // use LK flow to estimate points in the right image
    kps_right = kps_left;
    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame->image_left, current_frame->image_right, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // triangulate new points
    int num_triangulated_pts = 0;
    int num_good_pts = 0;
    for (size_t i = 0; i < kps_left.size(); ++i)
    {
        if (status[i])
        {
            num_good_pts++;
            // triangulation
            Vector2d kp_left = cv2eigen(kps_left[i]);
            Vector2d kp_right = cv2eigen(kps_right[i]);
            Vector3d pb = Vector3d::Zero();
            triangulate(camera_left_->extrinsic.inverse(), camera_right_->extrinsic.inverse(),
                        camera_left_->Pixel2Sensor(kp_left), camera_right_->Pixel2Sensor(kp_right), pb);
            if ((camera_left_->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (camera_right_->Robot2Pixel(pb) - kp_right).norm() < 0.5)
            {
                auto new_landmark = visual::Landmark::Create(pb, camera_left_);
                auto new_left_feature = visual::Feature::Create(current_frame, eigen2cv(kp_left), new_landmark);
                auto new_right_feature = visual::Feature::Create(current_frame, eigen2cv(kp_right), new_landmark);
                new_right_feature->is_on_left_image = false;
                new_landmark->AddObservation(new_left_feature);
                new_landmark->AddObservation(new_right_feature);
                current_frame->AddFeature(new_left_feature);
                current_frame->AddFeature(new_right_feature);
                map_->InsertLandmark(new_landmark);
                position_cache_[new_landmark->id] = new_landmark->ToWorld();
                num_triangulated_pts++;
            }
        }
    }

    //LOG(INFO) << "Detect " << kps_left.size() << " new features";
   // LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    //LOG(INFO) << "new landmarks: " << num_triangulated_pts;
    return num_triangulated_pts;
}

// TODO
bool Frontend::Reset()
{
    backend_.lock()->Pause();
    map_->Reset();
    backend_.lock()->Continue();
    status = FrontendStatus::BUILDING;
//NEWADD
    current_frame=Frame::Create();
    last_frame=Frame::Create();
    last_key_frame=static_cast<Frame::Ptr>(NULL);
    current_key_frame=static_cast<Frame::Ptr>(NULL);
//NEWADDEND
    //LOG(INFO) << "Reset Succeed";
    return true;
}

void Frontend::UpdateCache()
{
    position_cache_.clear();
    for (auto pair_feature : last_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        position_cache_[camera_point->id] = camera_point->ToWorld();
    }
    last_frame_pose_cache_ = last_frame->pose;
}

//NEWADD
void Frontend::UpdateFrameIMU(const double s, const Bias &b, Frame::Ptr pCurrentKeyFrame)
{
    last_key_frame = pCurrentKeyFrame;

    last_frame->SetNewBias(b);
    current_frame->SetNewBias(b);

    Vector3d Gz ;
    Gz << 0, 0, -ImuCalib_.G_norm;
    Gz =backend_.lock()->initializer_->mRwg*Gz;

    Vector3d twb1;
    Matrix3d Rwb1;
   Vector3d Vwb1;
    double t12;  // 时间间隔

    while(!current_frame->preintegration->isPreintegrated)
    {
        usleep(500);
    }

    // Step 2:更新Frame的pose velocity
    // Step 2.1:更新lastFrame的pose velocity
    if(last_frame->mpLastKeyFrame){
        if(last_frame->id== last_frame->mpLastKeyFrame->id)
        {
            last_frame->SetImuPoseVelocity(last_frame->mpLastKeyFrame->GetImuRotation(),
                                        last_frame->mpLastKeyFrame->GetImuPosition(),
                                        last_frame->mpLastKeyFrame->GetVelocity());
        }
        else
        {
            twb1 = last_frame->mpLastKeyFrame->GetImuPosition();
            Rwb1 = last_frame->mpLastKeyFrame->GetImuRotation();
            Vwb1 = last_frame->mpLastKeyFrame->GetVelocity();
            t12 = last_frame->preintegration->dT;

            last_frame->SetImuPoseVelocity(Rwb1*last_frame->preintegration->GetUpdatedDeltaRotation(),
                                        twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*last_frame->preintegration->GetUpdatedDeltaPosition(),
                                        Vwb1 + Gz*t12 + Rwb1*last_frame->preintegration->GetUpdatedDeltaVelocity());
        }
    }
    // Step 2.2:更新currentFrame的pose velocity
    if (current_frame->preintegration&&current_frame->mpLastKeyFrame)
    {
        twb1 = current_frame->mpLastKeyFrame->GetImuPosition();
        Rwb1 = current_frame->mpLastKeyFrame->GetImuRotation();
        Vwb1 = current_frame->mpLastKeyFrame->GetVelocity();
        t12 = current_frame->preintegration->dT;

        current_frame->SetImuPoseVelocity(Rwb1*current_frame->preintegration->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_frame->preintegration->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*current_frame->preintegration->GetUpdatedDeltaVelocity());
    }
}
//NEWADDEND
} // namespace lvio_fusion