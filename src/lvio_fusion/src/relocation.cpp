#include "relocation.h"
#include <DBoW3/QueryResults.h>

namespace lvio_fusion
{

Relocation::Relocation(std::string voc_path)
{
    thread_ = std::thread(std::bind(&Relocation::RelocationLoop, this));
    detector_ = cv::ORB::create();
    voc_ = DBoW3::Vocabulary(voc_path);
    db_ = DBoW3::Database(voc_, false, 0);
    head_ = 0;
}

void Relocation::RelocationLoop()
{
    while (true)
    {
        while (head_ < frames_.size())
        {
            AddKeyFrameIntoVoc(frames_[head_]);
            if(DetectLoop(frames_[head_]))
            {
                Associate(frames_[head_]);
            }
            head_++;
        }
        std::chrono::milliseconds dura(100);
        std::this_thread::sleep_for(dura);
    }
}

void Relocation::AddKeyFrameIntoVoc(Frame::Ptr frame)
{
    static int thershold = 20;
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(frame->image_left, keypoints, 20, true);
    detector_->compute(frame->image_left, keypoints, frame->descriptors);
    db_.add(frame->descriptors);
}

bool Relocation::DetectLoop(Frame::Ptr frame)
{
    //first query; then add this frame into database!
    DBoW3::QueryResults ret;
    db_.query(frame->descriptors, ret, 4, frame->id - 20);
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;
    // a good match with its nerghbour
    if (ret.size() >= 1 && ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
            }
        }
    
    if (find_loop && frame->id > 20)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        frame->loop = frames_[min_index];
        return true;
    }
    return false;
}

void Relocation::Associate(Frame::Ptr frame)
{
    Frame::Ptr base_frame = frame->loop;
    VisualAssociate();
    // use LK flow to estimate points in the last image
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    for (auto feature_pair : last_frame->features_left)
    {
        // use project point
        auto feature = feature_pair.second;
        auto camera_point = feature->landmark.lock();
        auto px = camera_left_->World2Pixel(position_cache_[camera_point->id], current_frame->pose);
        landmarks.push_back(camera_point);
        kps_last.push_back(eigen2cv(feature->keypoint));
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
            auto feature = visual::Feature::Create(current_frame, cv2eigen(kps_current[i]), landmarks[i]);
            current_frame->AddFeature(feature);
            num_good_pts++;
        }
    }
    cv::imshow("tracking", img_track);
    cv::waitKey(1);
    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;


    InitFramePoseByPnP();
}

bool Frontend::InitFramePoseByPnP()
{
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto feature_pair : current_frame->features_left)
    {
        auto feature = feature_pair.second;
        auto camera_point = feature->landmark.lock();
        points_2d.push_back(eigen2cv(feature->keypoint));
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
        current_frame->pose = camera_left_->extrinsic.inverse() *
                              SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
        return true;
    }
    return false;
}


} // namespace lvio_fusion