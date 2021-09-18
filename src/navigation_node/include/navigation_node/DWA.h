#ifndef navigation_node_DWA_H
#define navigation_node_DWA_H
#include "lvio_fusion/common.h"
#include <ros/ros.h>
#include <tf/tf.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <Eigen/Dense>
namespace navigation_node
{
   class State
    {
    public:
        State(double _x, double _y, double _yaw, double _velocity, double _yawrate)
            :x(_x), y(_y), yaw(_yaw), velocity(_velocity), yawrate(_yawrate)
        {
        }

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
       Window(void)
        :min_velocity(0.0), max_velocity(0.0), min_yawrate(0.0), max_yawrate(0.0)
        {
        }

        Window(const double min_v, const double max_v, const double min_y, const double max_y)
            :min_velocity(min_v), max_velocity(max_v), min_yawrate(min_y), max_yawrate(max_y)
        {
        }
        double min_velocity;
        double max_velocity;
        double min_yawrate;
        double max_yawrate;
    private:
    };
    
    class DWA
    {
    public:
        typedef std::shared_ptr<DWA> Ptr;
        DWA(){ }

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
                    float final_cost = TO_GOAL_COST_GAIN*to_goal_cost + SPEED_COST_GAIN*speed_cost + OBSTACLE_COST_GAIN*obstacle_cost;
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
            // ros::Rate loop_rate(HZ);

            while(ros::ok()){
            //     ROS_INFO("==========================================");
            //     double start = ros::Time::now().toSec();
            bool input_updated = false;
                if(USE_SCAN_AS_INPUT && scan_updated){
                    input_updated = true;
                }else if(!USE_SCAN_AS_INPUT && local_map_updated){
                    input_updated = true;
                }
            if(input_updated && local_goal_subscribed && odom_updated){
                Window dynamic_window = calc_dynamic_window(current_velocity);
                Eigen::Vector3d goal(local_goal.pose.position.x, local_goal.pose.position.y, tf::getYaw(local_goal.pose.orientation));
            //         ROS_INFO_STREAM("local goal: (" << goal[0] << "," << goal[1] << "," << goal[2]/M_PI*180 << ")");

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
                        // visualize_trajectory(best_traj, 1, 0, 0, selected_trajectory_pub);
                    }else{
                        cmd_vel.linear.x = 0.0;
                        if(fabs(goal[2])>TURN_DIRECTION_THRESHOLD){
                            cmd_vel.angular.z = std::min(std::max(goal(2), -MAX_YAWRATE), MAX_YAWRATE);
                        }
                        else{
                            cmd_vel.angular.z = 0.0;
                            //need next goal
                        }
                    }
            //         ROS_INFO_STREAM("cmd_vel: (" << cmd_vel.linear.x << "[m/s], " << cmd_vel.angular.z << "[rad/s])");
                    velocity_pub.publish(cmd_vel);

            //         odom_updated = false;
                 }
                 //else{
            //         if(!local_goal_subscribed){
            //             ROS_WARN_THROTTLE(1.0, "Local goal has not been updated");
            //         }
            //         if(!odom_updated){
            //             ROS_WARN_THROTTLE(1.0, "Odom has not been updated");
            //         }
            //         if(!USE_SCAN_AS_INPUT && !local_map_updated){
            //             ROS_WARN_THROTTLE(1.0, "Local map has not been updated");
            //         }
            //         if(USE_SCAN_AS_INPUT && !scan_updated){
            //             ROS_WARN_THROTTLE(1.0, "Scan has not been updated");
            //         }
            //     }
            //     ros::spinOnce();
            //     loop_rate.sleep();
            //     ROS_INFO_STREAM("loop time: " << ros::Time::now().toSec() - start << "[s]");
            }
        }
        Window calc_dynamic_window(const geometry_msgs::Twist& current_velocity)
        {
            Window window(MIN_VELOCITY, MAX_VELOCITY, -MAX_YAWRATE, MAX_YAWRATE);
            window.min_velocity = std::max((current_velocity.linear.x - MAX_ACCELERATION*DT), MIN_VELOCITY);
            window.max_velocity = std::min((current_velocity.linear.x + MAX_ACCELERATION*DT), MAX_VELOCITY);
            window.min_yawrate = std::max((current_velocity.angular.z - MAX_D_YAWRATE*DT), -MAX_YAWRATE);
            window.max_yawrate = std::min((current_velocity.angular.z + MAX_D_YAWRATE*DT),  MAX_YAWRATE);
            return window;
        }

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

        std::vector<std::vector<float>> scan_to_obs()
        {
            std::vector<std::vector<float>> obs_list;
            float angle = scan.angle_min;
            for(auto r : scan.ranges){
                float x = r * cos(angle);
                float y = r * sin(angle);
                std::vector<float> obs_state = {x, y};
                obs_list.push_back(obs_state);
                angle += scan.angle_increment;
            }
            return obs_list;
        }

        std::vector<std::vector<float>> raycast()
        {
            Vector2d robot_position_;
            double yaw_;
            std::vector<std::vector<float>> obs_list;
            for(float angle = -M_PI; angle <= M_PI; angle += ANGLE_RESOLUTION){
                for(float dist = 0.0; dist <= MAX_DIST; dist += local_map.info.resolution){
                    float x = dist * cos(angle+yaw_)+robot_position_[0];
                    float y = dist * sin(angle+yaw_)+robot_position_[1];
                    int i = floor(x / local_map.info.resolution + 0.5) + local_map.info.width * 0.5;
                    int j = floor(y / local_map.info.resolution + 0.5) + local_map.info.height * 0.5;
                    if( (i < 0 || i >= local_map.info.width) || (j < 0 || j >= local_map.info.height) ){
                        break;
                    }
                    if(local_map.data[j*local_map.info.width + i] == 100){
                        std::vector<float> obs_state = {x, y};
                        obs_list.push_back(obs_state);
                        break;
                    }
                }
            }
            return obs_list;
        }

        void set_local_goal(const geometry_msgs::PoseStampedConstPtr& msg)
        {
            local_goal = *msg;
            try{
                listener.transformPose(ROBOT_FRAME, ros::Time(0), local_goal, local_goal.header.frame_id, local_goal);
                local_goal_subscribed = true;
            }catch(tf::TransformException ex){
                ROS_ERROR("%s", ex.what());
            }
        }

        void set_scan(const sensor_msgs::LaserScanConstPtr& msg)
        {
            scan = *msg;
            scan_updated = true;
        }

        void set_local_map(const nav_msgs::OccupancyGridConstPtr& msg)
        {
            local_map = *msg;
            local_map_updated = true;
        }

        void set_odom(const nav_msgs::OdometryConstPtr& msg)
        {
            current_velocity = msg->twist.twist;
            odom_updated = true;
        }

        protected:
            double HZ;
            std::string ROBOT_FRAME;
            double TARGET_VELOCITY;
            double MAX_VELOCITY;
            double MIN_VELOCITY;
            double MAX_YAWRATE;
            double MAX_ACCELERATION;
            double MAX_D_YAWRATE;
            double MAX_DIST;
            double VELOCITY_RESOLUTION;
            double YAWRATE_RESOLUTION;
            double ANGLE_RESOLUTION;
            double PREDICT_TIME;
            double TO_GOAL_COST_GAIN;
            double SPEED_COST_GAIN;
            double OBSTACLE_COST_GAIN;
            double DT;
            bool USE_SCAN_AS_INPUT;
            double GOAL_THRESHOLD;
            double TURN_DIRECTION_THRESHOLD;

            ros::NodeHandle nh;
            ros::NodeHandle local_nh;

            ros::Publisher velocity_pub;
            ros::Publisher candidate_trajectories_pub;
            ros::Publisher selected_trajectory_pub;
            ros::Subscriber local_map_sub;
            ros::Subscriber scan_sub;
            ros::Subscriber local_goal_sub;
            ros::Subscriber odom_sub;
            ros::Subscriber target_velocity_sub;
            tf::TransformListener listener;

            geometry_msgs::PoseStamped local_goal;
            sensor_msgs::LaserScan scan;
            nav_msgs::OccupancyGrid local_map;
            geometry_msgs::Twist current_velocity;
            bool local_goal_subscribed;
            bool scan_updated;
            bool local_map_updated;
            bool odom_updated;
     };
}// namespace navigation_node
#endif // navigation_node_DWA_H