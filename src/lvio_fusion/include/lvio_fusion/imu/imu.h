#ifndef lvio_fusion_IMU_H
#define lvio_fusion_IMU_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

struct ImuData
{
    Vector3d a; //acceleration
    Vector3d w;//angular velocity
    double t;     //timestamp

    ImuData(const double &acc_x, const double &acc_y, const double &acc_z,
             const double &ang_vel_x, const double &ang_vel_y, const double &ang_vel_z,
             const double &timestamp) : a(acc_x, acc_y, acc_z), w(ang_vel_x, ang_vel_y, ang_vel_z), t(timestamp) {}
    ImuData(const Vector3d Acc, const Vector3d Gyro, const double &timestamp) : a(Acc[0], Acc[1], Acc[2]), w(Gyro[0], Gyro[1], Gyro[2]), t(timestamp) {}
    ImuData()
    {
        a = Vector3d::Zero();
        w = Vector3d::Zero();
        t = 0;
    }
};

struct Bias
{
    Vector3d linearized_ba;//accelerometer bias 
    Vector3d linearized_bg;//gyroscope bias

    
    Bias() : linearized_ba(Vector3d::Zero()), linearized_bg(Vector3d::Zero()) {}
    Bias(Vector3d linearized_ba_, Vector3d linearized_bg_) : linearized_ba(linearized_ba_), linearized_bg(linearized_bg_) {}
    Bias(double b_acc_x, double b_acc_y, double b_acc_z,
         double b_ang_vel_x, double b_ang_vel_y, double b_ang_vel_z)
    {
        linearized_ba << b_acc_x, b_acc_y, b_acc_z;
        linearized_bg << b_ang_vel_x, b_ang_vel_y, b_ang_vel_z;
    }
};

class Imu : public Sensor
{
public:
    typedef std::shared_ptr<Imu> Ptr;

    static int Create(const SE3d &extrinsic, double acc_n, double acc_w, double gyr_n, double gyr_w, double g_norm)
    {
        devices_.push_back(Imu::Ptr(new Imu(extrinsic, acc_n, acc_w, gyr_n, gyr_w, g_norm)));
        return devices_.size() - 1;
    }

    static int Num()
    {
        return devices_.size();
    }

    static Imu::Ptr Get(int id = 0)
    {
        return devices_[id];
    }

    SE3d GetPose_G(SE3d pose)
    {
        return SE3d(Rwg.inverse() * pose.rotationMatrix(), Rwg.inverse() * pose.translation());
    }
    
    double ACC_N, ACC_W;    //noise and  random walk
    double GYR_N, GYR_W;
    double G;                               //value of Gravity
    Matrix3d Rwg;                      //Gravity direction
    bool initialized = false;

private:
    Imu(const SE3d &extrinsic, double acc_n, double acc_w, double gyr_n, double gyr_w, double g_norm) : Sensor(extrinsic), ACC_N(acc_n), ACC_W(acc_w), GYR_N(gyr_n), GYR_W(gyr_w), G(g_norm) { Rwg = Matrix3d::Identity(); }
    Imu(const Imu &);
    Imu &operator=(const Imu &);

    static std::vector<Imu::Ptr> devices_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_H
