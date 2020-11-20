#ifndef lvio_fusion_IMU_H
#define lvio_fusion_IMU_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
namespace lvio_fusion
{
const double eps = 1e-4;
class Imu : public Sensor
{
public:
    typedef std::shared_ptr<Imu> Ptr;

    Imu(const SE3d &extrinsic) : Sensor(extrinsic) {}

    double ACC_N, ACC_W;
    double GYR_N, GYR_W;
};
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
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & bax;
        ar & bay;
        ar & baz;

        ar & bwx;
        ar & bwy;
        ar & bwz;
    }

public:
    Bias():bax(0),bay(0),baz(0),bwx(0),bwy(0),bwz(0){}
    Bias(double b_acc_x, double b_acc_y,double b_acc_z,
            double b_ang_vel_x, double b_ang_vel_y, double b_ang_vel_z):
            bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z){}
    void CopyFrom(Bias &b)
    {
    bax = b.bax;
    bay = b.bay;
    baz = b.baz;
    bwx = b.bwx;
    bwy = b.bwy;
    bwz = b.bwz;
    }
    friend std::ostream& operator<< (std::ostream &out, const Bias &b)
    {
    if(b.bwx>0)
        out << " ";
    out << b.bwx << ",";
    if(b.bwy>0)
        out << " ";
    out << b.bwy << ",";
    if(b.bwz>0)
        out << " ";
    out << b.bwz << ",";
    if(b.bax>0)
        out << " ";
    out << b.bax << ",";
    if(b.bay>0)
        out << " ";
    out << b.bay << ",";
    if(b.baz>0)
        out << " ";
    out << b.baz;

    return out;
    }
    double bax=0, bay=0, baz=0;
    double bwx=0, bwy=0, bwz=0;
};

class IntegratedRotation
{
public:
    IntegratedRotation(){}
    IntegratedRotation(const Vector3d &angVel, const Bias &imuBias, const double &time)
{
    const double x = (angVel[0]-imuBias.bwx)*time;
    const double y = (angVel[1]-imuBias.bwy)*time;
    const double z = (angVel[2]-imuBias.bwz)*time;

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

class Calib
{
    template<class Archive>
    void serializeMatrix(Archive &ar, cv::Mat& mat, const unsigned int version)
    {
        int cols, rows, type;
        bool continuous;

        if (Archive::is_saving::value) {
            cols = mat.cols; rows = mat.rows; type = mat.type();
            continuous = mat.isContinuous();
        }

        ar & cols & rows & type & continuous;
        if (Archive::is_loading::value)
            mat.create(rows, cols, type);

        if (continuous) {
            const unsigned int data_size = rows * cols * mat.elemSize();
            ar & boost::serialization::make_array(mat.ptr(), data_size);
        } else {
            const unsigned int row_size = cols*mat.elemSize();
            for (int i = 0; i < rows; i++) {
                ar & boost::serialization::make_array(mat.ptr(i), row_size);
            }
        }
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        serializeMatrix(ar,Tcb,version);
        serializeMatrix(ar,Tbc,version);
        serializeMatrix(ar,Cov,version);
        serializeMatrix(ar,CovWalk,version);
    }

public:
    Calib(const Matrix4d &Tbc_, const double &ng, const double &na, const double &ngw, const double &naw,const double g_norm_)
    {
       G_norm=g_norm_;
        Set(Tbc_,ng,na,ngw,naw);
    }
    Calib(const Calib &calib)
    {
    Tbc = calib.Tbc;
    Tcb = calib.Tcb;
    Cov = calib.Cov;
    CovWalk = calib.CovWalk;
    G_norm=calib.G_norm;
    }
    Calib(){}

    void Set(const Matrix4d &Tbc_, const double &ng, const double &na, const double &ngw, const double &naw)
    {
    Tbc = Tbc_;
    Tcb = Matrix4d::Identity();
    Tcb.block<3, 3>(0,0) = Tbc.block<3, 3>(0,0).transpose();
    Tcb.block<3, 1>(0,3) = -Tbc.block<3, 3>(0,0).transpose()*Tbc.block<3, 1>(0,3);
    Cov = Matrix<double, 6, 6> ::Identity();
    const double ng2 = ng*ng;
    const double na2 = na*na;
    Cov(0,0) = ng2;
    Cov(1,1) = ng2;
    Cov(2,2) = ng2;
    Cov(3,3) = na2;
    Cov(4,4) = na2;
    Cov(5,5) = na2;
    CovWalk = Matrix<double, 6, 6> ::Identity();
    const double ngw2 = ngw*ngw;
    const double naw2 = naw*naw;
    CovWalk(0,0) = ngw2;
    CovWalk(1,1) = ngw2;
    CovWalk(2,2) = ngw2;
    CovWalk(3,3) = naw2;
    CovWalk(4,4) = naw2;
    CovWalk(5,5) = naw2;
    }
    
public:
    Matrix4d Tcb;  //b to camera
    Matrix4d Tbc;
    Matrix<double, 6, 6> Cov, CovWalk; // imu协方差， 随机游走协方差
    double G_norm;
};


} // namespace lvio_fusion
#endif // lvio_fusion_IMU_H
