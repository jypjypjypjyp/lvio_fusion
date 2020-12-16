#ifndef lvio_fusion_IMU_H
#define lvio_fusion_IMU_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
namespace lvio_fusion
{
const double eps = 1e-4;
//NEWADD
class Imu : public Sensor
{
public:
    typedef std::shared_ptr<Imu> Ptr;

    Imu(const SE3d &extrinsic, double acc_n, double  acc_w, double gyr_n,double gyr_w,double g_norm) : Sensor(extrinsic),ACC_N(acc_n),ACC_W(acc_w),GYR_N(gyr_n),GYR_W(gyr_w), G(g_norm){}


    double ACC_N, ACC_W;
    double GYR_N, GYR_W;
    double G;
    bool initialized =false;
};
//NEWADDEND
class  imuPoint
{
public:
    typedef std::shared_ptr<imuPoint> Ptr;
    imuPoint(const double &acc_x, const double &acc_y, const double &acc_z,
             const double &ang_vel_x, const double &ang_vel_y, const double &ang_vel_z,
             const double &timestamp): a(acc_x,acc_y,acc_z), w(ang_vel_x,ang_vel_y,ang_vel_z), t(timestamp){}
    imuPoint(const Vector3d Acc, const  Vector3d Gyro, const double &timestamp):
             a(Acc[0],Acc[1],Acc[2]), w(Gyro[0],Gyro[1],Gyro[2]), t(timestamp){}
public:
    Vector3d a;
    Vector3d w;
    double t;
};


class Bias
{
public:
    Bias():linearized_ba(Vector3d::Zero()),linearized_bg(Vector3d::Zero()){}
    Bias(Vector3d linearized_ba_,Vector3d linearized_bg_ ):linearized_ba(linearized_ba_),linearized_bg(linearized_bg_){}
    Bias(double b_acc_x, double b_acc_y,double b_acc_z,
            double b_ang_vel_x, double b_ang_vel_y, double b_ang_vel_z)
            {
                linearized_ba<< b_acc_x,  b_acc_y,  b_acc_z;
                linearized_bg<< b_ang_vel_x,  b_ang_vel_y,  b_ang_vel_z;
            }
    Vector3d linearized_ba;
    Vector3d linearized_bg;
};

class IntegratedRotation
{
public:
    IntegratedRotation(){}
    IntegratedRotation(const Vector3d &angVel, const Bias &imuBias, const double &time)
{
    const double x = (angVel[0]-imuBias.linearized_bg[0])*time;
    const double y = (angVel[1]-imuBias.linearized_bg[1])*time;
    const double z = (angVel[2]-imuBias.linearized_bg[2])*time;

    Matrix3d I =Matrix3d::Identity();

    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Matrix3d W;
    W << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0;
    if(d<eps)
    {
        deltaR = I + W;                                    // 公式(4)
        rightJ = Matrix3d::Identity();
    }
    else
    {
        deltaR = I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2;   //罗德里格斯 公式(3)
        rightJ = I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);   //公式(8)
    }
}

public:
    double deltaT; //integration time
    Matrix3d deltaR; //integrated rotation
    Matrix3d rightJ; // right jacobian
};

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_H
