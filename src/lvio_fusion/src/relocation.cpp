#include "lvio_fusion/loop/relocation.h"
#include "lvio_fusion/ceres/imu_error.hpp"
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
    static double end_time = DBL_MAX;
    static double start_time = 0;
    static double head = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto new_kfs = map_->GetKeyFrames(head, map_->time_local_map);
        if (new_kfs.empty())
        {
            continue;
        }
        for (auto pair_kf : new_kfs)
        {
            Frame::Ptr frame = pair_kf.second, frame_old;
            AddKeyFrameIntoVoc(frame);
            // if last is loop and this is not loop, then correct all new loops
            bool is_loop = DetectLoop(frame, frame_old) && Associate(frame, frame_old);
            if (is_loop)
            {
                start_time = frame->time;
                end_time = std::min(end_time, frame_old->time);
                RefineAssociation(frame, frame_old);
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
    map_dbow_to_frames_.insert(std::make_pair(id, frame->time));
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
        frame_old = map_->GetAllKeyFrames()[map_dbow_to_frames_[min_index]];
        return true;
    }
    return false;
}

bool Relocation::Associate(Frame::Ptr frame, Frame::Ptr frame_old)
{
    loop::LoopConstraint::Ptr loop_constraint = loop::LoopConstraint::Ptr(new loop::LoopConstraint());
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2d> points_2d;
    // search by BRIEFDes
    auto descriptors = mat2briefs(frame);
    auto descriptors_old = mat2briefs(frame_old);
    for (auto pair_desciptor : descriptors)
    {
        unsigned long best_id = 0;
        if (SearchInAera(pair_desciptor.second, descriptors_old, best_id))
        {
            cv::Point2d point_2d = frame_old->features_left[best_id]->keypoint;
            visual::Landmark::Ptr landmark = frame_old->features_left[best_id]->landmark.lock();
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
        loop_constraint->frame_old = frame_old;
        // TODO
        if (RefineAssociation(frame, frame_old, loop_constraint))
        {
            frame->loop_constraint = loop_constraint;
            return true;
        }
    }
    return false;
}

bool Relocation::RefineAssociation(Frame::Ptr frame, Frame::Ptr frame_old, loop::LoopConstraint::Ptr loop_constraint)
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

void Relocation::BuildProblem(Frames &active_kfs, ceres::Problem &problem)
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
            auto first_frame = landmark->FirstFrame();
            ceres::CostFunction *cost_function;
            if (first_frame.lock()->time < start_time)
            {
                cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_kf);
            }
            else if (first_frame.lock() != frame)
            {
                double *para_fist_kf = first_frame.lock()->pose.data();
                cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), camera_left_);
                problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
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

    // imu constraints
    if (imu_ && imu_->initialized)
    {
        Frame::Ptr last_frame;
        Frame::Ptr current_frame;
        for (auto pair_kf : active_kfs)
        {
            current_frame = pair_kf.second;
            if (!current_frame->preintegration)
                continue;
            auto para_kf = current_frame->pose.data();
            auto para_v = current_frame->preintegration->v0.data();
            auto para_ba = current_frame->preintegration->linearized_ba.data();
            auto para_bg = current_frame->preintegration->linearized_bg.data();
            problem.AddParameterBlock(para_v, 3);
            problem.AddParameterBlock(para_ba, 3);
            problem.AddParameterBlock(para_bg, 3);
            if (last_frame && last_frame->preintegration)
            {
                auto para_kf_last = last_frame->pose.data();
                auto para_v_last = last_frame->preintegration->v0.data();
                auto para_ba_last = last_frame->preintegration->linearized_ba.data();
                auto para_bg_last = last_frame->preintegration->linearized_bg.data();
                ceres::CostFunction *cost_function = ImuError::Create(last_frame->preintegration);
                problem.AddResidualBlock(cost_function, NULL, para_kf_last, para_v_last, para_ba_last, para_bg_last, para_kf, para_v, para_ba, para_bg);
            }
            last_frame = current_frame;
        }
    }

    // loop constraints
    problem.SetParameterBlockConstant(active_kfs.begin()->second->pose.data());
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        if (frame->loop_constraint)
        {
            auto frame_old = frame->loop_constraint->frame_old;
            if (frame_old->time <= start_time)
            {
                problem.SetParameterBlockConstant(para_kf);
            }
            else
            {
                for (auto pair_feature : frame->loop_constraint->features_loop)
                {
                    auto feature = pair_feature.second;
                    auto landmark = feature->landmark.lock();
                    auto first_frame = landmark->FirstFrame();
                    ceres::CostFunction *cost_function;
                    if (first_frame.lock()->time < start_time)
                    {
                        cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), landmark->ToWorld(), camera_left_, 5);
                        problem.AddResidualBlock(cost_function, loss_function, para_kf);
                    }
                    else if (first_frame.lock() != frame)
                    {
                        double *para_fist_kf = first_frame.lock()->pose.data();
                        cost_function = TwoFrameReprojectionError::Create(landmark->position, cv2eigen(feature->keypoint), camera_left_, 5);
                        problem.AddResidualBlock(cost_function, loss_function, para_fist_kf, para_kf);
                    }
                }
            }
        }
    }
}

// TODO
void Relocation::CorrectLoop(double start_time, double end_time)
{
    Frames active_kfs = map_->GetKeyFrames(start_time, end_time);
    // stop mapping
    mapping_->Pause();

    // optimize pose graph
    ceres::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 5;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // reject outliers and clean the map
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        auto left_features = frame->features_left;
        for (auto pair_feature : left_features)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            auto first_frame = landmark->FirstFrame();
            Vector2d error(0, 0);
            if (first_frame.lock()->time < active_kfs.begin()->first)
            {
                PoseOnlyReprojectionError(cv2eigen(feature->keypoint), landmark->ToWorld(), camera_left_)(para_kf, error.data());
            }
            else if (first_frame.lock() != frame)
            {
                double *para_fist_kf = first_frame.lock()->pose.data();
                TwoFrameReprojectionError(landmark->position, cv2eigen(feature->keypoint), camera_left_)(para_fist_kf, para_kf, error.data());
            }
            if (error.norm() > 3)
            {
                landmark->RemoveObservation(feature);
                frame->RemoveFeature(feature);
            }
            if (landmark->observations.size() == 1 && frame->id != Frame::current_frame_id)
            {
                map_->RemoveLandmark(landmark);
            }
        }
    }

    // mapping
    mapping_->Optimize(start_time);

    // forward propogate
    {
        std::unique_lock<std::mutex> lock1(frontend_.lock()->mutex);
        std::unique_lock<std::mutex> lock2(backend_.lock()->mutex);

        Frame::Ptr last_frame = frontend_.lock()->last_frame;
        Frames active_kfs = map_->GetKeyFrames(end_time);
        if (active_kfs.find(last_frame->time) == active_kfs.end())
        {
            active_kfs.insert(std::make_pair(last_frame->time, last_frame));
        }
        Frame::Ptr frame = (--active_kfs.end())->second;
        Frame::Ptr frame_old = frame->loop_constraint->frame_old;
        SE3d transform_pose = frame->pose.inverse() * frame->loop_constraint->relative_pose * frame_old->pose;
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
}

} // namespace lvio_fusion