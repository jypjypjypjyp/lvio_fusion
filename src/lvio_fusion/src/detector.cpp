#include "lvio_fusion/loop/detector.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

#include <DBoW3/QueryResults.h>
#include <opencv2/core/eigen.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

namespace lvio_fusion
{

LoopDetector::LoopDetector(std::string voc_path)
{
    detector_ = cv::ORB::create();
    voc_ = DBoW3::Vocabulary(voc_path);
    db_ = DBoW3::Database(voc_, false, 0);
    thread_ = std::thread(std::bind(&LoopDetector::DetectorLoop, this));
}

void LoopDetector::DetectorLoop()
{
    static double old_time = DBL_MAX;
    static double start_time = 0;
    static Frame::Ptr last_frame;
    static Frame::Ptr last_old_frame;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto new_kfs = Map::Instance().GetKeyFrames(head, backend_->head);
        if (new_kfs.empty())
        {
            continue;
        }
        for (auto pair_kf : new_kfs)
        {
            Frame::Ptr frame = pair_kf.second, old_frame;
            AddKeyFrameIntoVoc(frame);
            // if last is loop and this is not loop, then correct all new loops
            if (DetectLoop(frame, old_frame))
            {
                if (!last_old_frame)
                {
                    start_time = pair_kf.first;
                }
                old_time = std::min(old_time, old_frame->time);
                last_frame = frame;
                last_old_frame = old_frame;
            }
            else if (start_time != DBL_MAX)
            {
                LOG(INFO) << "Detected new loop, and correct it now. old_time:" << old_time << ";start_time:" << start_time << ";end_time:" << last_frame->time;
                auto t1 = std::chrono::steady_clock::now();
                CorrectLoop(old_time, start_time, last_frame->time);
                auto t2 = std::chrono::steady_clock::now();
                auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
                LOG(INFO) << "Correct Loop cost time: " << time_used.count() << " seconds.";
                start_time = old_time = DBL_MAX;
                last_old_frame = last_frame = nullptr;
            }
        }
        head = (--new_kfs.end())->first + epsilon;
    }
}

void LoopDetector::AddKeyFrameIntoVoc(Frame::Ptr frame)
{
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    for (auto pair_feature : frame->features_left)
    {
        keypoints.push_back(cv::KeyPoint(pair_feature.second->keypoint, 1));
    }
    cv::Mat descriptors;
    detector_->compute(frame->image_left, keypoints, descriptors);
    DBoW3::EntryId id = db_.add(descriptors);
    map_dbow_to_frames_[id] = frame->time;

    // NOTE: detector_->compute maybe remove some row because its descriptor cannot be computed
    int j = 0, i = 0;
    frame->descriptors = cv::Mat::zeros(frame->features_left.size(), 32, CV_8U);
    for (auto pair_feature : frame->features_left)
    {
        if (pair_feature.second->keypoint == keypoints[j].pt && j < descriptors.rows)
        {
            descriptors.row(j).copyTo(frame->descriptors.row(i));
            j++;
        }
        i++;
    }
}

bool LoopDetector::DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame)
{
    // NOTE: DBow3 is not good
    // //first query; then add this frame into database!
    // DBoW3::QueryResults ret;
    // db_.query(frame->descriptors, ret, 4, frame->id - 20);
    // // ret[0] is the nearest neighbour's score. threshold change with neighour score
    // bool find_loop = false;
    // cv::Mat loop_result;
    // // a good match with its nerghbour
    // if (ret.size() >= 1 && ret[0].Score > 0.05)
    //     for (unsigned int i = 1; i < ret.size(); i++)
    //     {
    //         if (ret[i].Score > 0.015)
    //         {
    //             find_loop = true;
    //         }
    //     }
    // if (find_loop && frame->id > 20)
    // {
    //     int max_index = -1;
    //     for (unsigned int i = 0; i < ret.size(); i++)
    //     {
    //         if (max_index == -1 || (ret[i].Id > max_index && ret[i].Score > 0.015))
    //             max_index = ret[i].Id;
    //     }
    //     old_frame = Map::Instance().keyframes[map_dbow_to_frames_[max_index]];
    //     // check the distance
    //     if ((frame->pose.inverse().translation() - old_frame->pose.inverse().translation()).norm() < 20)
    //     {
    //         return true;
    //     }
    // }
    // return false;
    Frames candidate_kfs = Map::Instance().GetKeyFrames(0, backend_->head - 30);
    double min_distance = 10;
    for (auto pair_kf : candidate_kfs)
    {
        Vector3d vec = (pair_kf.second->pose.translation() - frame->pose.translation());
        vec.z() = 0;
        double distance = vec.norm();
        if (distance < min_distance)
        {
            Frame::Ptr prev_frame = Map::Instance().GetKeyFrames(0, frame->time, 1).begin()->second;
            Frame::Ptr subs_frame = Map::Instance().GetKeyFrames(frame->time, 0, 1).begin()->second;
            Vector3d prev_vec = (pair_kf.second->pose.translation() - prev_frame->pose.translation());
            Vector3d subs_vec = (pair_kf.second->pose.translation() - subs_frame->pose.translation());
            prev_vec.z() = 0;
            subs_vec.z() = 0;
            double prev_distance = prev_vec.norm();
            double subs_distance = subs_vec.norm();
            if (prev_distance < min_distance && subs_distance < min_distance)
            {
                min_distance = distance;
                old_frame = pair_kf.second;
            }
        }
    }
    if (old_frame)
    {
        loop::LoopClosure::Ptr loop_constraint = loop::LoopClosure::Ptr(new loop::LoopClosure());
        loop_constraint->frame_old = old_frame;
        loop_constraint->relocated = false;
        frame->loop_closure = loop_constraint;
        return true;
    }
    return false;
}

bool LoopDetector::Relocate(Frame::Ptr frame, Frame::Ptr old_frame)
{
    frame->loop_closure->score = 0;
    double rpyxyz_o[6], rpyxyz_i[6], rpy_o_i[3];
    se32rpyxyz(frame->pose, rpyxyz_i);
    se32rpyxyz(old_frame->pose, rpyxyz_o);
    rpy_o_i[0] = rpyxyz_i[0] - rpyxyz_o[0];
    rpy_o_i[1] = rpyxyz_i[1] - rpyxyz_o[1];
    rpy_o_i[2] = rpyxyz_i[2] - rpyxyz_o[2];
    //TODO
    // if (Vector3d(rpy_o_i[0], rpy_o_i[1], rpy_o_i[2]).norm() < 0.1)
    // {
    //     RelocateByImage(frame, old_frame);
    // }
    if (mapping_)
    {
        RelocateByPoints(frame, old_frame);
    }
    if (frame->loop_closure->score > 0)
    {
        return true;
    }
    frame->loop_closure.reset();
    return false;
}

bool LoopDetector::RelocateByImage(Frame::Ptr frame, Frame::Ptr old_frame)
{
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    // search by BRIEFDes
    auto descriptors = mat2briefs(frame);
    auto descriptors_old = mat2briefs(old_frame);
    for (auto pair_desciptor : descriptors)
    {
        unsigned long best_id = 0;
        if (SearchInAera(pair_desciptor.second, descriptors_old, best_id))
        {
            cv::Point2f point_2d = old_frame->features_left[best_id]->keypoint;
            visual::Landmark::Ptr landmark = old_frame->features_left[best_id]->landmark.lock();
            visual::Feature::Ptr new_left_feature = visual::Feature::Create(frame, point_2d, landmark);
            points_2d.push_back(point_2d);
            points_3d.push_back(eigen2cv(landmark->position));
        }
    }
    // solve pnp ransca
    cv::Mat K;
    cv::eigen2cv(Camera::Get()->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (points_2d.size() >= 20 && cv::solvePnPRansac(points_3d, points_2d, K, D, rvec, tvec, false, 100, 8.0F, 0.98, cv::noArray(), cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        frame->loop_closure->relative_o_c = (Camera::Get()->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();
        frame->loop_closure->score = std::min((double)points_2d.size(), 50.0);
        return true;
    }
    return false;
}

bool LoopDetector::RelocateByPoints(Frame::Ptr frame, Frame::Ptr old_frame)
{
    if (!frame->feature_lidar || !old_frame->feature_lidar)
    {
        return false;
    }

    // init relative pose
    Frame::Ptr clone_frame = Frame::Ptr(new Frame());
    *clone_frame = *frame;
    if (clone_frame->loop_closure->score > 0)
    {
        clone_frame->pose = clone_frame->loop_closure->relative_o_c * old_frame->pose;
    }
    else
    {
        clone_frame->pose.translation().z() = old_frame->pose.translation().z();
    }

    // build two pointclouds
    Frame::Ptr old_frame_prev = Map::Instance().GetKeyFrames(0, old_frame->time, 1).begin()->second;
    Frame::Ptr old_frame_subs = Map::Instance().GetKeyFrames(old_frame->time, 0, 1).begin()->second;
    Frames old_frames = {{old_frame->time, old_frame}, {old_frame_prev->time, old_frame_prev}, {old_frame_subs->time, old_frame_subs}};
    Frame::Ptr map_frame = Frame::Ptr(new Frame());
    mapping_->BuildOldMapFrame(old_frames, map_frame);
    // PointICloud points_temp_frame_world;
    // mapping_->MergeScan(clone_frame->feature_lidar->points_full, clone_frame->pose, points_temp_frame_world);
    // // save
    // pcl::io::savePCDFile("/home/zoet/Projects.new/lvio_fusion/result/" + std::to_string(frame->time) + ".pcd", points_temp_frame_world);
    // pcl::io::savePCDFile("/home/zoet/Projects.new/lvio_fusion/result/old_" + std::to_string(frame->time) + ".pcd", map_frame->feature_lidar->points_full);
    // // icp
    // pcl::IterativeClosestPoint<PointI, PointI> icp;
    // icp.setInputSource(boost::make_shared<PointICloud>(points_temp_frame_world));
    // icp.setInputTarget(boost::make_shared<PointICloud>(map_frame->feature_lidar->points_full));
    // icp.setMaximumIterations(100);
    // icp.setMaxCorrespondenceDistance(5);
    // PointICloud::Ptr aligned(new PointICloud);
    // icp.align(*aligned);
    // if (!icp.hasConverged() || icp.getFitnessScore() > 10)
    // {
    //     return false;
    // }
    // Matrix4d transform_matrix = icp.getFinalTransformation().cast<double>();
    // Matrix3d R(transform_matrix.block(0, 0, 3, 3));
    // Quaterniond q(R);
    // Vector3d t(0, 0, 0);
    // t << transform_matrix(0, 3), transform_matrix(1, 3), transform_matrix(2, 3);
    // SE3d transform(q, t);
    // clone_frame->pose = transform * clone_frame->pose;

    // optimize
    double score_ground, score_surf;
    for (int i = 0; i < 4; i++)
    {
        double rpyxyz[6];
        se32rpyxyz(clone_frame->pose * map_frame->pose.inverse(), rpyxyz); // relative_i_j
        if (!map_frame->feature_lidar->points_ground.empty())
        {
            adapt::Problem problem;
            association_->ScanToMapWithGround(clone_frame, map_frame, rpyxyz, problem);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.num_threads = 4;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            clone_frame->pose = rpyxyz2se3(rpyxyz) * map_frame->pose;
            score_ground = std::min((double)summary.num_residual_blocks_reduced / 10, 20.0);
            score_ground -= 2 * summary.final_cost / summary.num_residual_blocks_reduced;
        }
        if (!map_frame->feature_lidar->points_surf.empty())
        {
            adapt::Problem problem;
            association_->ScanToMapWithSegmented(clone_frame, map_frame, rpyxyz, problem);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.num_threads = 4;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            clone_frame->pose = rpyxyz2se3(rpyxyz) * map_frame->pose;
            score_surf = std::min((double)summary.num_residual_blocks_reduced / 10, 30.0);
            score_surf -= 2 * summary.final_cost / summary.num_residual_blocks_reduced;
        }
    }

    frame->loop_closure->relative_o_c = clone_frame->pose * old_frame->pose.inverse();
    frame->loop_closure->score += score_ground + score_surf;
    return true;
}

bool LoopDetector::SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id)
{
    cv::Point2f best_pt;
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

int LoopDetector::Hamming(const BRIEF &a, const BRIEF &b)
{
    BRIEF xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void LoopDetector::BuildProblem(Frames &active_kfs, adapt::Problem &problem)
{
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame;
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        double *para_kf = frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        if (last_frame)
        {
            double *para_last_kf = last_frame->pose.data();
            ceres::CostFunction *cost_function;
            cost_function = PoseGraphError::Create(last_frame->pose, frame->pose, frame->weights.pose_graph);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_last_kf, para_kf);
        }
        last_frame = frame;
    }
}

void LoopDetector::BuildProblemWithLoop(Frames &active_kfs, adapt::Problem &problem)
{
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;

    // loop constraint
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        if (frame->loop_closure && frame->loop_closure->relocated)
        {
            double *para_kf = frame->pose.data();
            problem.SetParameterBlockConstant(para_kf);

            auto old_frame = frame->loop_closure->frame_old;
            if (old_frame->time >= start_time)
            {
                double *para_old_kf = old_frame->pose.data();
                problem.SetParameterBlockConstant(para_old_kf);
            }
        }
    }
}

void LoopDetector::CorrectLoop(double old_time, double start_time, double end_time)
{
    // build the pose graph and submaps
    Frames active_kfs = Map::Instance().GetKeyFrames(old_time, end_time);
    Frames new_submap_kfs = Map::Instance().GetKeyFrames(start_time, end_time);
    Frames all_kfs = active_kfs;
    // std::map<double, SE3d> inner_submap_old_frames = atlas_.GetActiveSubMaps(active_kfs, old_time, start_time);
    // atlas_.AddSubMap(old_time, start_time, end_time);
    // adapt::Problem problem;
    // BuildProblem(active_kfs, problem);

    // update new submap frams
    SE3d old_pose = (--new_submap_kfs.end())->second->pose;
    {
        // build new submap pose graph
        adapt::Problem problem;
        BuildProblem(new_submap_kfs, problem);

        // relocate new submaps
        std::map<double, double> score_table;
        for (auto pair_kf : new_submap_kfs)
        {
            Relocate(pair_kf.second, pair_kf.second->loop_closure->frame_old);
            score_table[-pair_kf.second->loop_closure->score] = pair_kf.first;
        }
        int max_num_relocated = 1;
        for (auto pair : score_table)
        {
            if (max_num_relocated-- == 0)
                break;
            auto frame = new_submap_kfs[pair.second];
            frame->loop_closure->relocated = true;
            frame->pose = frame->loop_closure->relative_o_c * frame->loop_closure->frame_old->pose;
        }

        BuildProblemWithLoop(new_submap_kfs, problem);
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        for (auto pair_kf : new_submap_kfs)
        {
            Relocate(pair_kf.second, pair_kf.second->loop_closure->frame_old);
            pair_kf.second->loop_closure->relocated = true;
            pair_kf.second->pose = pair_kf.second->loop_closure->relative_o_c * pair_kf.second->loop_closure->frame_old->pose;
        }
    }
    SE3d new_pose = (--new_submap_kfs.end())->second->pose;

    // forward propogate
    {
        std::unique_lock<std::mutex> lock1(backend_->mutex);
        std::unique_lock<std::mutex> lock2(frontend_->mutex);

        Frame::Ptr last_frame = frontend_->last_frame;
        Frames forward_kfs = Map::Instance().GetKeyFrames(end_time + epsilon);
        if (forward_kfs.find(last_frame->time) == forward_kfs.end())
        {
            forward_kfs[last_frame->time] = last_frame;
        }
        SE3d transform = old_pose.inverse() * new_pose;
        for (auto pair_kf : forward_kfs)
        {
            pair_kf.second->pose = pair_kf.second->pose * transform;
            // TODO: Repropagate

            if (mapping_)
            {
                mapping_->ToWorld(pair_kf.second);
            }
        }
        frontend_->UpdateCache();
    }

    // BuildProblemWithLoop(active_kfs, problem);
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.num_threads = 1;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);

    // // update pose of inner submaps
    // for (auto pair_of : inner_submap_old_frames)
    // {
    //     auto old_frame = active_kfs[pair_of.first];
    //     // T2_new = T2 * T1.inverse() * T1_new
    //     SE3d transform = pair_of.second.inverse() * old_frame->pose;
    //     for (auto iter = ++all_kfs.find(pair_of.first); active_kfs.find(iter->first) == active_kfs.end(); iter++)
    //     {
    //         auto frame = iter->second;
    //         frame->pose = frame->pose * transform;
    //     }
    // }

    // if (mapping_)
    // {
    //     for (auto pair_kf : all_kfs)
    //     {
    //         mapping_->AddToWorld(pair_kf.second);
    //     }
    // }
}

} // namespace lvio_fusion