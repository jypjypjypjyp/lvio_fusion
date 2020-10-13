#include "lvio_fusion/loop/relocation.h"
#include "lvio_fusion/utility.h"

#include <DBoW3/QueryResults.h>
#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Relocation::Relocation(std::string voc_path)
{
    thread_ = std::thread(std::bind(&Relocation::RelocationLoop, this));
    detector_ = cv::ORB::create();
    voc_ = DBoW3::Vocabulary(voc_path);
    db_ = DBoW3::Database(voc_, false, 0);
}

void Relocation::UpdateMap()
{
    map_update_.notify_one();
}

void Relocation::Pause()
{
    if (status == RelocationStatus::RUNNING)
    {
        std::unique_lock<std::mutex> lock(pausing_mutex_);
        status = RelocationStatus::TO_PAUSE;
        pausing_.wait(lock);
    }
}

void Relocation::Continue()
{
    if (status == RelocationStatus::PAUSING)
    {
        status = RelocationStatus::RUNNING;
        running_.notify_one();
    }
}

void Relocation::RelocationLoop()
{
    static bool is_last_loop = false;
    static double end_time = DBL_MAX;
    static double start_time = 0;
    static double head = 0;
    while (true)
    {
        std::unique_lock<std::mutex> lock(running_mutex_);
        if (status == RelocationStatus::TO_PAUSE)
        {
            status = RelocationStatus::PAUSING;
            pausing_.notify_one();
            running_.wait(lock);
        }
        map_update_.wait(lock);
        auto new_kfs = map_->GetKeyFrames(head);
        for (auto kf_pair : new_kfs)
        {
            Frame::Ptr frame = kf_pair.second, old_frame;
            AddKeyFrameIntoVoc(frame);
            bool is_loop = DetectLoop(frame, old_frame) && Associate(frame, old_frame);
            if (is_loop)
            {
                start_time = frame->time;
                end_time = std::min(end_time, old_frame->time);
            }
            else if (is_last_loop)
            {
                CorrectLoop(start_time, end_time);
                end_time = DBL_MAX;
                start_time = 0;
            }
            is_last_loop = is_loop;
        }
        head = (--new_kfs.end())->first;
        std::chrono::milliseconds dura(100);
        std::this_thread::sleep_for(dura);
    }
}

void Relocation::AddKeyFrameIntoVoc(Frame::Ptr frame)
{
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    for (auto pair : frame->features_left)
    {
        keypoints.push_back(cv::KeyPoint(pair.second->keypoint, 1));
    }
    detector_->compute(frame->image_left, keypoints, frame->descriptors);
    LOG(INFO) << frame->descriptors;
    DBoW3::EntryId id = db_.add(frame->descriptors);
    map_db_to_frames_.insert(std::make_pair(id, frame->time));
}

bool Relocation::DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame)
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
        old_frame = map_->GetAllKeyFrames()[map_db_to_frames_[min_index]];
        return true;
    }
    return false;
}

bool Relocation::Associate(Frame::Ptr frame, Frame::Ptr &old_frame)
{
    loop::LoopConstraint::Ptr loop_constraint = loop::LoopConstraint::Ptr(new loop::LoopConstraint());
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2d> points_2d;
    // search by BRIEFDes
    auto descriptors = mat2briefs(frame);
    auto descriptors_old = mat2briefs(old_frame);
    for (auto pair : descriptors)
    {
        unsigned long best_id = 0;
        if (SearchInAera(pair.second, descriptors_old, best_id))
        {
            cv::Point2d point_2d = old_frame->features_left[best_id]->keypoint;
            visual::Landmark::Ptr landmark = old_frame->features_left[best_id]->landmark.lock();
            visual::Feature::Ptr new_left_feature = visual::Feature::Create(frame, point_2d, landmark);
            loop_constraint->features_loop.insert(std::make_pair(landmark->id, new_left_feature));
            points_2d.push_back(point_2d);
            points_3d.push_back(eigen2cv(landmark->position));
        }
    }
    // solve pnp ransca
    cv::Mat K;
    cv::eigen2cv(camera_left_->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (cv::solvePnPRansac(points_3d, points_2d, K, D, rvec, tvec, false, 100, 8.0F, 0.98, cv::noArray(), cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        loop_constraint->relative_pose = camera_left_->extrinsic.inverse() * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
        loop_constraint->frame_old = old_frame;
        frame->pose = loop_constraint->relative_pose * old_frame->pose;
        frame->loop_constraint = loop_constraint;
        return true;
    }
    return false;
}

bool Relocation::SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id)
{
    cv::Point2d best_pt;
    int best_distance = 256;
    for (auto pair : descriptors_old)
    {
        int distance = Hamming(descriptor, pair.second);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_id = pair.first;
        }
    }
    return best_distance < 160;
}

int Relocation::Hamming(const BRIEF &a, const BRIEF &b)
{
    BRIEF xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

// TODO
void Relocation::CorrectLoop(double start_time, double end_time)
{
    double relocation_head = relocation_.lock()->head;
    Frames frames = map_->GetKeyFrames(global_head, relocation_head);
    Frame::Ptr frame, old_frame;
    bool has_loop = false;
    for (auto kf_pair : frames)
    {
        if (kf_pair.second->loop_constraint)
        {
            has_loop = true;
            frame = kf_pair.second;
            old_frame = frame->loop_constraint->frame_old;
            break;
        }
    }
    global_head = relocation_head;
    if (has_loop)
    {
        // Correct Loop
        {
            // forward
            std::unique_lock<std::mutex> lock(running_mutex_);
            std::unique_lock<std::mutex> lock(frontend_.lock()->last_frame_mutex);

            Frame::Ptr last_frame = frontend_.lock()->last_frame;
            Frames active_kfs = map_->GetKeyFrames(old_frame->time);
            if (active_kfs.find(last_frame->time) == active_kfs.end())
            {
                active_kfs.insert(std::make_pair(last_frame->time, last_frame));
            }
            SE3d transform_pose = frame->pose.inverse() * frame->loop_constraint->relative_pose * old_frame->pose;
            for (auto kf_pair : active_kfs)
            {
                kf_pair.second->pose = kf_pair.second->pose * transform_pose;
                // TODO: Repropagate
                // if(kf_pair.second->preintegration)
                // {
                //     kf_pair.second->preintegration->Repropagate();
                // }
            }
            frontend_.lock()->UpdateCache();
        }
    }
}
} // namespace lvio_fusion