#include "lvio_fusion/loop/relocation.h"
#include "lvio_fusion/loop/loop_constraint.h"
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
    head = 0;
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
            Frame::Ptr frame = kf_pair.second, frame_old;
            AddKeyFrameIntoVoc(frame);
            if (DetectLoop(frame, frame_old))
            {
                Associate(frame, frame_old);
            }
            head = kf_pair.first;
        }
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

bool Relocation::DetectLoop(Frame::Ptr frame, Frame::Ptr &frame_old)
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
        frame_old = map_->GetAllKeyFrames()[map_db_to_frames_[min_index]];
        return true;
    }
    return false;
}

void Relocation::Associate(Frame::Ptr frame, Frame::Ptr &frame_old)
{
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2d> points_2d;
    SearchByBRIEFDes(frame, frame_old, points_3d, points_2d);
    if (UpdateFramePoseByPnP(frame, points_3d, points_2d))
    {
        if (lidar_)
        {
            UpdateFramePoseByLidar(frame, frame_old);
        }
        loop::LoopConstraint::Ptr loop_constraint = loop::LoopConstraint::Ptr(new loop::LoopConstraint());
        loop_constraint->relative_pose = frame->pose * frame_old->pose.inverse();
        loop_constraint->frame_old = frame_old;
        frame->loop_constraint = loop_constraint;
    }
}

void Relocation::SearchByBRIEFDes(Frame::Ptr frame, Frame::Ptr frame_old, std::vector<cv::Point3d> &points_3d, std::vector<cv::Point2d> &points_2d)
{
    auto descriptors = mat2briefs(frame);
    auto descriptors_old = mat2briefs(frame_old);
    for (auto pair : descriptors)
    {
        unsigned long best_id = 0;
        if (SearchInAera(pair.second, descriptors_old, best_id))
        {
            points_2d.push_back(frame_old->features_left[best_id]->keypoint);
            points_3d.push_back(eigen2cv(frame_old->features_left[best_id]->landmark.lock()->position));
        }
    }
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

bool Relocation::UpdateFramePoseByPnP(Frame::Ptr frame, std::vector<cv::Point3d> points_3d, std::vector<cv::Point2d> points_2d)
{
    cv::Mat K;
    cv::eigen2cv(camera_left_->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (cv::solvePnPRansac(points_3d, points_2d, K, D, rvec, tvec, false, 100, 8.0F, 0.98, cv::noArray(), cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        frame->pose = camera_left_->extrinsic.inverse() *
                      SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
        return true;
    }
    return false;
}

void Relocation::UpdateFramePoseByLidar(Frame::Ptr frame, Frame::Ptr frame_old)
{
    ceres::Problem problem;
    ceres::LossFunction *lidar_loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double *para_kf = frame->pose.data();
    problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
    double *para_kf_old = frame_old->pose.data();
    problem.AddParameterBlock(para_kf_old, SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(para_kf_old);

    scan_registration_->Associate(frame, frame_old, problem, lidar_loss_function);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.function_tolerance = 1e-9;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

} // namespace lvio_fusion