#include "lvio_fusion/loop/relocation.h"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

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

void Relocation::RelocationLoop()
{
    static bool is_last_loop = false;
    static double old_time = DBL_MAX;
    static double end_time = 0;
    static double start_time = 0;
    static double last_time = 0;
    static double head = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto new_kfs = map_->GetKeyFrames(head, map_->local_map_head);
        if (new_kfs.empty())
        {
            continue;
        }
        for (auto pair_kf : new_kfs)
        {
            Frame::Ptr frame = pair_kf.second, old_frame;
            AddKeyFrameIntoVoc(frame);
            // if last is loop and this is not loop, then correct all new loops
            bool is_loop = DetectLoop(frame, old_frame) && Associate(frame, old_frame);
            if (is_loop)
            {
                if (!is_last_loop)
                {
                    // get all new loop frames (start_time, end_time]
                    start_time = last_time;
                }
                end_time = frame->time;
                old_time = std::min(end_time, old_frame->time);
            }
            else if (is_last_loop)
            {
                CorrectLoop(old_time, start_time, end_time);
                old_time = DBL_MAX;
                end_time = 0;
            }
            is_last_loop = is_loop;
            last_time = pair_kf.first;
        }
        head = (--new_kfs.end())->first;
    }
}

void Relocation::AddKeyFrameIntoVoc(Frame::Ptr frame)
{
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    for (auto pair_feature : frame->features_left)
    {
        keypoints.push_back(cv::KeyPoint(pair_feature.second->keypoint, 1));
    }
    detector_->compute(frame->image_left, keypoints, frame->descriptors);
    LOG(INFO) << frame->descriptors;
    DBoW3::EntryId id = db_.add(frame->descriptors);
    map_dbow_to_frames_[id] = frame->time;
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
        int max_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (max_index == -1 || (ret[i].Id > max_index && ret[i].Score > 0.015))
                max_index = ret[i].Id;
        }
        old_frame = map_->GetAllKeyFrames()[map_dbow_to_frames_[max_index]];
        return true;
    }
    return false;
}

bool Relocation::Associate(Frame::Ptr frame, Frame::Ptr old_frame)
{
    loop::LoopConstraint::Ptr loop_constraint = loop::LoopConstraint::Ptr(new loop::LoopConstraint());
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2d> points_2d;
    // search by BRIEFDes
    auto descriptors = mat2briefs(frame);
    auto descriptors_old = mat2briefs(old_frame);
    for (auto pair_desciptor : descriptors)
    {
        unsigned long best_id = 0;
        if (SearchInAera(pair_desciptor.second, descriptors_old, best_id))
        {
            cv::Point2d point_2d = old_frame->features_left[best_id]->keypoint;
            visual::Landmark::Ptr landmark = old_frame->features_left[best_id]->landmark.lock();
            visual::Feature::Ptr new_left_feature = visual::Feature::Create(frame, point_2d, landmark);
            loop_constraint->features_loop[landmark->id] = new_left_feature;
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
        if (lidar_ && RefineAssociation(frame, old_frame, loop_constraint))
        {
            frame->loop_constraint = loop_constraint;
            return true;
        }
    }
    return false;
}

bool Relocation::RefineAssociation(Frame::Ptr frame, Frame::Ptr old_frame, loop::LoopConstraint::Ptr loop_constraint)
{
    Frame frame_copy = *frame;
    Frame::Ptr frame_temp = Frame::Ptr(&frame_copy);
    frame_temp->pose = loop_constraint->relative_pose * old_frame->pose;
    static int num_iters = 2;
    for (int i = 0; i < num_iters; i++)
    {
        ceres::Problem problem;
        ceres::LossFunction *lidar_loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));

        double *para_kf = frame_temp->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        double *para_kf_old = old_frame->pose.data();
        problem.AddParameterBlock(para_kf_old, SE3d::num_parameters, local_parameterization);
        problem.SetParameterBlockConstant(para_kf_old);

        scan_registration_->Associate(frame_temp, old_frame, problem, lidar_loss_function);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 3;
        options.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
    loop_constraint->relative_pose = frame_temp->pose * old_frame->pose.inverse();
    return true;
}

bool Relocation::SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id)
{
    cv::Point2d best_pt;
    int best_distance = 256;
    for (auto pair_desciptor : descriptors_old)
    {
        int distance = Hamming(descriptor, pair_desciptor.second);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_id = pair_desciptor.first;
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

void Relocation::BuildProblem(Frames &active_kfs, std::map<double, SE3d> &inner_old_frame, ceres::Problem &problem)
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
            if (first_frame->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else if (first_frame != frame)
            {
                if (active_kfs.find(first_frame->time) == active_kfs.end())
                {
                    auto old_frame = active_kfs[inner_old_frame.upper_bound(first_frame->time)->first];
                    SE3d relative_pose = first_frame->pose * old_frame->pose.inverse();
                    double *para_old_kf = old_frame->pose.data();
                    cost_function = TwoFrameReprojectionErrorBasedLoop::Create(landmark->position, cv2eigen(feature->keypoint), camera_left_, relative_pose);
                    problem.AddResidualBlock(cost_function, loss_function, para_old_kf, para_kf);
                }
                else
                {
                    double *para_fist_kf = first_frame->pose.data();
                    cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), camera_left_);
                    problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
                }
            }
        }
    }

    // navsat constraints
    auto navsat_map = map_->navsat_map;
    if (map_->navsat_map != nullptr && navsat_map->initialized)
    {
        ceres::LossFunction *navsat_loss_function = new ceres::TrivialLoss();
        for (auto pair_kf : active_kfs)
        {
            auto frame = pair_kf.second;
            auto para_kf = frame->pose.data();
            auto np_iter = navsat_map->navsat_points.lower_bound(pair_kf.first);
            auto navsat_point = np_iter->second;
            navsat_map->Transfrom(navsat_point);
            if (std::fabs(navsat_point.time - frame->time) < 1e-2)
            {
                ceres::CostFunction *cost_function = NavsatError::Create(navsat_point.position);
                problem.AddResidualBlock(cost_function, navsat_loss_function, para_kf);
            }
        }
    }

    // TODO:
    // imu constraints
    // if (imu_ && imu_->initialized)
    // {
    //     Frame::Ptr last_frame;
    //     Frame::Ptr current_frame;
    //     for (auto pair_kf : active_kfs)
    //     {
    //         current_frame = pair_kf.second;
    //         if (!current_frame->preintegration)
    //             continue;
    //         auto para_kf = current_frame->pose.data();
    //         auto para_v = current_frame->preintegration->v0.data();
    //         auto para_ba = current_frame->preintegration->linearized_ba.data();
    //         auto para_bg = current_frame->preintegration->linearized_bg.data();
    //         problem.AddParameterBlock(para_v, 3);
    //         problem.AddParameterBlock(para_ba, 3);
    //         problem.AddParameterBlock(para_bg, 3);
    //         if (last_frame && last_frame->preintegration)
    //         {
    //             auto para_kf_last = last_frame->pose.data();
    //             auto para_v_last = last_frame->preintegration->v0.data();
    //             auto para_ba_last = last_frame->preintegration->linearized_ba.data();
    //             auto para_bg_last = last_frame->preintegration->linearized_bg.data();
    //             ceres::CostFunction *cost_function = ImuError::Create(last_frame->preintegration);
    //             problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
    //         }
    //         last_frame = current_frame;
    //     }
    // }

    // loop constraint
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        if(frame->loop_constraint)
        {
            double *para_kf = frame->pose.data();
            problem.SetParameterBlockConstant(para_kf);
        }
    }
}

void Relocation::CorrectLoop(double old_time, double start_time, double end_time)
{
    // stop mapping
    mapping_->Pause();

    // update pose of new submap
    Frames new_submap_kfs = map_->GetKeyFrames(start_time, end_time);
    for (auto pair_kf : new_submap_kfs)
    {
        Frame::Ptr frame = pair_kf.second;
        if (frame->loop_constraint)
        {
            frame->pose = frame->loop_constraint->relative_pose * frame->loop_constraint->frame_old->pose;
        }
    }

    // build the active submaps
    Frames active_kfs = map_->GetKeyFrames(old_time, end_time);
    std::map<double, SE3d> inner_old_frames = atlas_.GetActiveSubMaps(active_kfs, old_time, start_time, end_time);
    atlas_.AddSubMap(old_time, start_time, end_time);

    // optimize pose graph
    ceres::Problem problem;
    BuildProblem(active_kfs, inner_old_frames, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 5;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // mapping
    mapping_->Optimize(active_kfs);

    // update pose of inner submaps
    active_kfs = map_->GetKeyFrames(old_time, start_time);
    for (auto iter = active_kfs.begin(); iter != active_kfs.end();)
    {
        if (iter->first > start_time)
            break;
        Frame::Ptr frame = iter->second;
        if (frame->loop_constraint)
        {
            Frame::Ptr old_frame = frame->loop_constraint->frame_old;
            if (old_frame->time > old_time)
            {
                auto iter_old_frame = active_kfs.find(old_frame->time);
                iter = active_kfs.erase(++iter_old_frame, iter);
                continue;
            }
        }
        iter++;
    }

    // forward propogate
    {
        std::unique_lock<std::mutex> lock1(frontend_.lock()->mutex);
        std::unique_lock<std::mutex> lock2(backend_.lock()->mutex);

        Frame::Ptr last_frame = frontend_.lock()->last_frame;
        Frames active_kfs = map_->GetKeyFrames(end_time);
        if (active_kfs.find(last_frame->time) == active_kfs.end())
        {
            active_kfs[last_frame->time] = last_frame;
        }
        Frame::Ptr frame = (--active_kfs.end())->second;
        Frame::Ptr old_frame = frame->loop_constraint->frame_old;
        SE3d transform_pose = frame->pose.inverse() * frame->loop_constraint->relative_pose * old_frame->pose;
        for (auto pair_kf : active_kfs)
        {
            pair_kf.second->pose = pair_kf.second->pose * transform_pose;
            // TODO: Repropagate
            // if(pair_kf.second->preintegration)
            // {
            //     pair_kf.second->preintegration->Repropagate();
            // }
        }
        frontend_.lock()->UpdateCache();
    }

    // mapping start
    mapping_->Continue();
}

} // namespace lvio_fusion