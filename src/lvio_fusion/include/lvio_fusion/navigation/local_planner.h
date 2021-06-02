#ifndef lvio_fusion_LOCALPLANNER_H
#define lvio_fusion_LOCALPLANNER_H
#include "lvio_fusion/common.h"

namespace lvio_fusion
{
   class State
    {
    public:
        State(double, double, double, double, double);

        double x;// robot position x
        double y;// robot posiiton y
        double yaw;// robot orientation yaw
        double velocity;// robot linear velocity
        double yawrate;// robot angular velocity
    private:
    };

    class Window
    {
    public:
        Window(void);
        Window(const double, const double, const double, const double);
        double min_velocity;
        double max_velocity;
        double min_yawrate;
        double max_yawrate;
    private:
    };
    
    class Local_planner
    {
    public:
        typedef std::shared_ptr<Local_planner> Ptr;
        Local_planner(){

        }

std::vector<State> dwa_planning(
        Window dynamic_window, 
        Eigen::Vector3d goal,
        std::vector<std::vector<float>> obs_list)
{
    float min_cost = 1e6;
    float min_obs_cost = min_cost;
    float min_goal_cost = min_cost;
    float min_speed_cost = min_cost;

    std::vector<std::vector<State>> trajectories;
    std::vector<State> best_traj;

    for(float v=dynamic_window.min_velocity; v<=dynamic_window.max_velocity; v+=VELOCITY_RESOLUTION){
        for(float y=dynamic_window.min_yawrate; y<=dynamic_window.max_yawrate; y+=YAWRATE_RESOLUTION){
            State state(0.0, 0.0, 0.0, current_velocity.linear.x, current_velocity.angular.z);
            std::vector<State> traj;
            for(float t=0; t<=PREDICT_TIME; t+=DT){
                motion(state, v, y);
                traj.push_back(state);
                t += DT;
            }
            trajectories.push_back(traj);

            float to_goal_cost = calc_to_goal_cost(traj, goal);
            float speed_cost = calc_speed_cost(traj, TARGET_VELOCITY);
            float obstacle_cost = calc_obstacle_cost(traj, obs_list);
            float final_cost = TO_GOAL_COST_GAIN*to_goal_c                      
            GAIN*speed_cost + OBSTACLE_COST_GAIN*obstacle_cost;
            if(min_cost >= final_cost){
                min_goal_cost = TO_GOAL_COST_GAIN*to_goal_cost;
                min_obs_cost = OBSTACLE_COST_GAIN*obstacle_cost;
                min_speed_cost = SPEED_COST_GAIN*speed_cost;
                min_cost = final_cost;
                best_traj = traj;
            }
        }
    }
    if(min_cost == 1e6){
        std::vector<State> traj;
        State state(0.0, 0.0, 0.0, current_velocity.linear.x, current_velocity.angular.z);
        traj.push_back(state);
        best_traj = traj;
    }
    return best_traj;
}
void process(void)
{
    ros::Rate loop_rate(HZ);

    while(ros::ok()){
        ROS_INFO("==========================================");
        double start = ros::Time::now().toSec();
        bool input_updated = false;
        if(USE_SCAN_AS_INPUT && scan_updated){
            input_updated = true;
        }else if(!USE_SCAN_AS_INPUT && local_map_updated){
            input_updated = true;
        }
        if(input_updated && local_goal_subscribed && odom_updated){
            Window dynamic_window = calc_dynamic_window(current_velocity);
            Eigen::Vector3d goal(local_goal.pose.position.x, local_goal.pose.position.y, tf::getYaw(local_goal.pose.orientation));
            ROS_INFO_STREAM("local goal: (" << goal[0] << "," << goal[1] << "," << goal[2]/M_PI*180 << ")");

            geometry_msgs::Twist cmd_vel;
            if(goal.segment(0, 2).norm() > GOAL_THRESHOLD){
                std::vector<std::vector<float>> obs_list;
                if(USE_SCAN_AS_INPUT){
                    obs_list = scan_to_obs();
                    scan_updated = false;
                }else{
                    obs_list = raycast();
                    local_map_updated = false;
                }

                std::vector<State> best_traj = dwa_planning(dynamic_window, goal, obs_list);

                cmd_vel.linear.x = best_traj[0].velocity;
                cmd_vel.angular.z = best_traj[0].yawrate;
                visualize_trajectory(best_traj, 1, 0, 0, selected_trajectory_pub);
            }else{
                cmd_vel.linear.x = 0.0;
                if(fabs(goal[2])>TURN_DIRECTION_THRESHOLD){
                    cmd_vel.angular.z = std::min(std::max(goal(2), -MAX_YAWRATE), MAX_YAWRATE);
                }
                else{
                    cmd_vel.angular.z = 0.0;
                }
            }
            ROS_INFO_STREAM("cmd_vel: (" << cmd_vel.linear.x << "[m/s], " << cmd_vel.angular.z << "[rad/s])");
            velocity_pub.publish(cmd_vel);

            odom_updated = false;
        }else{
            if(!local_goal_subscribed){
                ROS_WARN_THROTTLE(1.0, "Local goal has not been updated");
            }
            if(!odom_updated){
                ROS_WARN_THROTTLE(1.0, "Odom has not been updated");
            }
            if(!USE_SCAN_AS_INPUT && !local_map_updated){
                ROS_WARN_THROTTLE(1.0, "Local map has not been updated");
            }
            if(USE_SCAN_AS_INPUT && !scan_updated){
                ROS_WARN_THROTTLE(1.0, "Scan has not been updated");
            }
        }
        ros::spinOnce();
        loop_rate.sleep();
        ROS_INFO_STREAM("loop time: " << ros::Time::now().toSec() - start << "[s]");
    }
}
    private:

float calc_to_goal_cost(const std::vector<State>& traj, const Eigen::Vector3d& goal)
{
    Eigen::Vector3d last_position(traj.back().x, traj.back().y, traj.back().yaw);
    return (last_position.segment(0, 2) - goal.segment(0, 2)).norm();
}

float calc_speed_cost(const std::vector<State>& traj, const float target_velocity)
{
    float cost = fabs(target_velocity - fabs(traj[traj.size()-1].velocity));
    return cost;
}

float calc_obstacle_cost(const std::vector<State>& traj, const std::vector<std::vector<float>>& obs_list)
{
    float cost = 0.0;
    float min_dist = 1e3;
    for(const auto& state : traj){
        for(const auto& obs : obs_list){
            float dist = sqrt((state.x - obs[0])*(state.x - obs[0]) + (state.y - obs[1])*(state.y - obs[1]));
            if(dist <= local_map.info.resolution){
                cost = 1e6;
                return cost;
            }
            min_dist = std::min(min_dist, dist);
        }
    }
    cost = 1.0 / min_dist;
    return cost;
}

void motion(State& state, const double velocity, const double yawrate)
{
    state.yaw += yawrate*DT;
    state.x += velocity*std::cos(state.yaw)*DT;
    state.y += velocity*std::sin(state.yaw)*DT;
    state.velocity = velocity;
    state.yawrate = yawrate;
}

};
} // namespace lvio_fusion
#endif // lvio_fusion_LOCALPLANNER_H