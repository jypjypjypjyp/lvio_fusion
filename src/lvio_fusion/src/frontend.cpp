#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/ceres/visual_error.hpp"
#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Frontend::Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe)
    : num_features_(num_features), num_features_init_(init), num_features_tracking_(tracking),
      num_features_tracking_bad_(tracking_bad), num_features_needed_for_keyframe_(need_for_keyframe)
{}

bool Frontend::AddFrame(lvio_fusion::Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex);
    //NEWADD
    if(imu_)
    {
        if(last_frame)
            frame->preintegration = imu::Preintegration::Create(last_frame->GetImuBias(),imu_);
        else
            frame->preintegration = imu::Preintegration::Create(frame->GetImuBias(),imu_);
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
        imuData_buf.push_back(imuMeas);
    }
    //NEWADDEND
}

bool Frontend::Track()
{
    
    //NEWADD
    if(imu_)
    {
        if(last_key_frame&&last_key_frame->preintegration)
        {
            if(backend_.lock()->initializer_->bimu)
            {
                current_frame->bImu=true;
            }
            std::vector<imuPoint> imuDatafromlastkeyframe;
            while(true)
            {
                if(imuData_buf.empty())
                {
                    usleep(500);
                    continue;
                }
                imuPoint imudata = imuData_buf.front();
                if(imudata.t<last_key_frame->time-0.001)
                {
                    imuData_buf.pop_front();
                }
                else if(imudata.t<current_frame->time-0.001)
                {
                    imuDatafromlastkeyframe.push_back(imudata);
                    imuData_buf.pop_front();
                }
                else
                {
                    imuDatafromlastkeyframe.push_back(imudata);
                    break;
                }
            }
            current_frame->preintegration->PreintegrateIMU(imuDatafromlastkeyframe,last_key_frame->time, current_frame->time);
            if(backend_.lock()->initializer_->initialized)
            {
                 PredictVelocity();
            }
        }
    }
    if(!imu_||!backend_.lock()->initializer_->initialized)
    {
        current_frame->pose = relative_i_j * last_frame_pose_cache_;
    }

    if(imu_)
    {
        if(current_key_frame)
       {
            current_frame->last_keyframe = current_key_frame;
            current_frame->SetNewBias(current_key_frame->GetImuBias());
       }
    }

    //NEWADDEND
    TrackLastFrame();
    InitFramePoseByPnP();
    //LocalBA();
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
    //NEWADD
    if(imu_)
    { 
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
    Map::Instance().InsertKeyFrame(current_frame);
    current_key_frame = current_frame;
    //NEWADD
    LOG(INFO)<<"KeyFrame "<<current_key_frame->id<<"    :"<<current_key_frame->pose.translation().transpose();


    //NEWADDEND

   // LOG(INFO) << "Add a keyframe " << current_frame->id;
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

//NEWADD
inline double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}


void mycalcOpticalFlowPyrLK( cv::Mat& prevImg, cv::Mat& nextImg,
                                        std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& nextPts,
                                        std::vector<uchar>& status,  cv::Mat& err )
                                        {
cv::calcOpticalFlowPyrLK(
        prevImg, nextImg, prevPts,
        nextPts, status, err, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

        {
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_pts = prevPts;
            cv::calcOpticalFlowPyrLK(nextImg, prevImg, nextPts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

            for(size_t i = 0; i < status.size(); i++)
            {
                if(status[i] && reverse_status[i] && distance(prevPts[i] , reverse_pts[i]) <= 0.5&&nextPts[i].x>=0&&nextPts[i].x<prevImg.cols&&nextPts[i].y>=0&&nextPts[i].y<prevImg.rows)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }
}

//NEWADDEND
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
    //NEWADD
    mycalcOpticalFlowPyrLK(
        last_frame->image_left, current_frame->image_left, kps_last,
        kps_current, status, error);
    //NEWADDEND
    // NOTE: don‘t use, accuracy is very low
    // cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, 3.0, 0.9, status);

    int num_good_pts = 0;
    //DEBUG
    mask_ = cv::Mat(current_frame->image_left.size(), CV_8UC1, cv::Scalar(255));//NEWADD
    cv::Mat img_track = current_frame->image_left;
    cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i] &&mask_.at<uchar>(kps_current[i]) == 255 )//NEWADD
        {
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            auto feature = visual::Feature::Create(current_frame, kps_current[i], landmarks[i]);
            current_frame->AddFeature(feature);
            cv::circle(mask_, feature->keypoint ,30, 0, cv::FILLED);//NEWADD
            num_good_pts++;
        }
    }
    cv::imshow("tracking", img_track);
    cv::waitKey(1);
  //  LOG(INFO) << "Find " << num_good_pts << " in the last image.";
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
    Map::Instance().InsertKeyFrame(current_frame);
    LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend and loop because we have a new keyframe
    backend_.lock()->UpdateMap();
    return true;
}


int Frontend::DetectNewFeatures()
{
    // cv::Mat mask(current_frame->image_left.size(), CV_8UC1, 255);
    // for (auto pair_feature : current_frame->features_left)
    // {
    //     auto feature = pair_feature.second;
    // }

    std::vector<cv::Point2f> kps_left, kps_right; // must be point2f
    cv::goodFeaturesToTrack(current_frame->image_left, kps_left, num_features_ - current_frame->features_left.size(), 0.01, 30, mask_);//NEWADD

    // use LK flow to estimate points in the right image
    kps_right = kps_left;
    std::vector<uchar> status;
    cv::Mat error;
    //NEWADD
    mycalcOpticalFlowPyrLK(
        current_frame->image_left, current_frame->image_right, kps_left,
        kps_right, status, error);
    //NEWADDEND
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
                Map::Instance().InsertLandmark(new_landmark);
                position_cache_[new_landmark->id] = new_landmark->ToWorld();
                num_triangulated_pts++;
            }
        }
    }

    LOG(INFO) << "Detect " << kps_left.size() << " new features";
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    LOG(INFO) << "new landmarks: " << num_triangulated_pts;
    return num_triangulated_pts;
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
    current_key_frame=static_cast<Frame::Ptr>(NULL);
    //NEWADDEND
    LOG(INFO) << "Reset Succeed";
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
void Frontend::UpdateFrameIMU(const double s, const Bias &bias_, Frame::Ptr CurrentKeyFrame)
{
    last_key_frame = CurrentKeyFrame;

    last_frame->SetNewBias(bias_);
    current_frame->SetNewBias(bias_);

    Vector3d Gz ;
    Gz << 0, 0, -imu_->G;
    Gz =backend_.lock()->initializer_->Rwg*Gz;

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
    if(last_frame->last_keyframe){
        if(last_frame->id== last_frame->last_keyframe->id)
        {
            last_frame->SetPose(last_frame->last_keyframe->GetImuRotation(),last_frame->last_keyframe->GetImuPosition());
            last_frame->SetVelocity(last_frame->last_keyframe->GetVelocity());
        }
        else
        {
            twb1= last_frame->last_keyframe->GetImuPosition();
            Rwb1 = last_frame->last_keyframe->GetImuRotation();
            Vwb1 = last_frame->last_keyframe->GetVelocity();
            t12 = last_frame->preintegration->dT;
            last_frame->SetPose(Rwb1*last_frame->preintegration->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*last_frame->preintegration->GetUpdatedDeltaPosition());
            last_frame->SetVelocity(Vwb1 + Gz*t12 + Rwb1*last_frame->preintegration->GetUpdatedDeltaVelocity());
        }
    }
    // Step 2.2:更新currentFrame的pose velocity
    if (current_frame->preintegration&&current_frame->last_keyframe)
    {
         twb1= current_frame->last_keyframe->GetImuPosition();
        Rwb1 = current_frame->last_keyframe->GetImuRotation();
        Vwb1 = current_frame->last_keyframe->GetVelocity();
        t12 = current_frame->preintegration->dT;
        current_frame->SetPose(Rwb1*current_frame->preintegration->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_frame->preintegration->GetUpdatedDeltaPosition());
        current_frame->SetVelocity( Vwb1 + Gz*t12 + Rwb1*current_frame->preintegration->GetUpdatedDeltaVelocity());
    }
}

void Frontend::PredictVelocity()
{
    if(last_key_frame)
    {
        Vector3d Gz ;
        Gz << 0, 0, -imu_->G;
        Gz =backend_.lock()->initializer_->Rwg*Gz;
        double t12=current_frame->preintegration->dT;
         Vector3d twb1=last_key_frame->GetImuPosition();
        Matrix3d Rwb1=last_key_frame->GetImuRotation();
        Vector3d Vwb1=last_key_frame->Vw;

        Matrix3d Rwb2=NormalizeRotation(Rwb1*current_frame->preintegration->GetDeltaRotation(last_key_frame->GetImuBias()));
        Vector3d twb2=twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*current_frame->preintegration->GetDeltaPosition(last_key_frame->GetImuBias());
        Vector3d Vwb2=Vwb1+t12*Gz+Rwb1*current_frame->preintegration->GetDeltaVelocity(current_frame->GetImuBias());
    
        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2,twb2);
    }
}
//  void Frontend::LocalBA(){
//      adapt::Problem problem;
//       ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
//           ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
//         new ceres::EigenQuaternionParameterization(),
//         new ceres::IdentityParameterization(3));
//       auto para_kf=current_frame->pose.data();
//       problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
//       for (auto pair_feature : current_frame->features_left)
//         {
//             auto feature = pair_feature.second;
//             auto landmark = feature->landmark.lock();
//             auto first_frame = landmark->FirstFrame().lock();
//             ceres::CostFunction *cost_function;
//             cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint),position_cache_[landmark->id], camera_left_, current_frame->weights.visual);
//             problem.AddResidualBlock(ProblemType::PoseOnlyReprojectionError, cost_function, loss_function, para_kf);
//         }
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_QR;
//     options.max_num_iterations = 1;
//     options.num_threads = 1;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//  }

//NEWADDEND
} // namespace lvio_fusion