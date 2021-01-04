#ifndef lvio_fusion_ENVIRONMENT_H
#define lvio_fusion_ENVIRONMENT_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/adapt/weights.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/map.h"

#include <random>

namespace lvio_fusion
{

struct VirtualFrame
{
    Frame::Ptr frame;
    SE3d pose;
    SE3d groundtruth;
    double time;
};

typedef std::map<double, VirtualFrame> VirtualFrames;

struct Observation
{
    cv::Mat image;
    PointICloud points_ground;
    PointICloud points_surf;
};

class Environment
{
public:
    typedef std::shared_ptr<Environment> Ptr;

    static void Step(int id, Weights *weights, Observation* obs, double *reward, bool *done)
    {
        environments_[id]->Step(weights, obs, reward, done);
    }

    static void Reset(int id)
    {
        environments_[id]->Reset();
    }

    static void Init()
    {
        if (!ground_truths.empty())
        {
            double start_time = Map::Instance().keyframes.begin()->first;
            auto iter = Map::Instance().keyframes.end();
            for (int i = 0; i < num_virtual_frames_; i++)
            {
                iter--;
            }
            double end_time = iter->first;
            u_ = std::uniform_real_distribution<double>(start_time, end_time);
            initialized_ = true;
        }
    }

    static int Create()
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!initialized_)
            return -1;
        environments_.push_back(Environment::Ptr(new Environment()));
        return environments_.size() - 1;
    }

    static SE3d GetGroundTruth(double time)
    {
        auto iter = ground_truths.lower_bound(time);

        if (iter == ground_truths.begin())
        {
            return iter->second;
        }
        else if (iter == ground_truths.end())
        {
            return (--iter)->second;
        }
        else
        {
            auto next_iter = iter;
            auto prev_iter = --iter;
            if (time - prev_iter->first < next_iter->first - time)
            {
                return prev_iter->second;
            }
            else
            {
                return next_iter->second;
            }
        }
    }

    static std::map<double, SE3d> ground_truths;
    static std::mutex mutex;

private:
    Environment()
    {
        double time = u_(e_);
        Frames active_kfs = Map::Instance().GetKeyFrames(time, 0, num_virtual_frames_);
        for (auto pair : active_kfs)
        {
            VirtualFrame virtual_frame;
            virtual_frame.frame = pair.second;
            virtual_frame.groundtruth = GetGroundTruth(pair.first);
            virtual_frame.pose = virtual_frame.groundtruth;
            virtual_env[pair.first] = virtual_frame;
        }
    }

    void Step(Weights *weights, Observation* obs, double *reward, bool *done);

    void Reset()
    {
        state_ = virtual_env.begin()->first;
    }

    void BuildProblem(adapt::Problem &problem);

    static std::default_random_engine e_;
    static std::uniform_real_distribution<double> u_;
    static std::vector<Environment::Ptr> environments_;
    static int num_virtual_frames_;
    static bool initialized_;
    
    VirtualFrames virtual_env;
    double state_;
    double id_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_ENVIRONMENT_H
