#ifndef lvio_fusion_ENVIRONMENT_H
#define lvio_fusion_ENVIRONMENT_H

#include "lvio_fusion/adapt/observation.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/adapt/weights.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/estimator.h"
#include "lvio_fusion/map.h"

#include <random>

namespace lvio_fusion
{

class Environment
{
public:
    typedef std::shared_ptr<Environment> Ptr;

    static void Step(int id, Weights &weights, Observation &obs, float *reward, bool *done)
    {
        environments_[id]->Step(weights, obs, reward, done);
    }

    static void Init(Estimator::Ptr estimator)
    {
        if (!ground_truths.empty() && estimator)
        {
            estimator_ = estimator;

            // initialize map with ground turth
            for (auto &pair : Map::Instance().keyframes)
            {
                pair.second->pose = GetGroundTruth(pair.first);
            }

            // initialize random distribution
            double start_time = (++Map::Instance().keyframes.begin())->first;
            double end_time = (--Map::Instance().keyframes.end())->first;
            u_ = std::uniform_real_distribution<double>(start_time, end_time);
            initialized_ = true;
        }
    }

    static int Create(Observation &obs)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!initialized_)
            return -1;
        environments_.push_back(Environment::Ptr(new Environment(true)));
        int id = environments_.size() - 1;
        obs = environments_[id]->state_->second->GetObservation();
        return id;
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

    // evaluate
    Environment()
    {
        state_ = Map::Instance().keyframes.begin();
    }

    void Step(Weights &weights, Observation &obs);

    static std::map<double, SE3d> ground_truths;
    static std::mutex mutex;

private:
    Environment(bool train)
    {
        std::default_random_engine e;
        double time = u_(e);
        frames_ = Map::Instance().GetKeyFrames(time, 0, num_frames_per_env_);
        state_ = frames_.begin();
    }

    void Step(Weights &weights, Observation &obs, float *reward, bool *done);

    SE3d Optimize();

    static std::uniform_real_distribution<double> u_;
    static std::vector<Environment::Ptr> environments_;
    static Estimator::Ptr estimator_;
    static int num_frames_per_env_;
    static bool initialized_;

    Frames frames_;
    Frames::iterator state_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_ENVIRONMENT_H
