#ifndef lvio_fusion_IMU_H
#define lvio_fusion_IMU_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
namespace lvio_fusion
{
const float eps = 1e-4;
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
    imuPoint(const float &acc_x, const float &acc_y, const float &acc_z,
             const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
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
    Bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z,
            const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z):
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


public:
    float bax, bay, baz;
    float bwx, bwy, bwz;
};

class IntegratedRotation
{
public:
    IntegratedRotation(){}
    IntegratedRotation(const Vector3d &angVel, const Bias &imuBias, const float &time)
{
    const float x = (angVel[0]-imuBias.bwx)*time;
    const float y = (angVel[1]-imuBias.bwy)*time;
    const float z = (angVel[2]-imuBias.bwz)*time;

    cv::Mat I = cv::Mat::eye(3,3,CV_32F);

    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);

    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0);
    if(d<eps)
    {
        deltaR = I + W;                                    // 公式(4)
        rightJ = cv::Mat::eye(3,3,CV_32F);
    }
    else
    {
        deltaR = I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2;   //罗德里格斯 公式(3)
        rightJ = I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);   //公式(8)
    }
}

public:
    float deltaT; //integration time
    cv::Mat deltaR; //integrated rotation
    cv::Mat rightJ; // right jacobian
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
    Calib(const cv::Mat &Tbc_, const float &ng, const float &na, const float &ngw, const float &naw)
    {
        Set(Tbc_,ng,na,ngw,naw);
    }
    Calib(const Calib &calib)
    {
    Tbc = calib.Tbc.clone();
    Tcb = calib.Tcb.clone();
    Cov = calib.Cov.clone();
    CovWalk = calib.CovWalk.clone();
    }
    Calib(){}

    void Set(const cv::Mat &Tbc_, const float &ng, const float &na, const float &ngw, const float &naw)
    {
    Tbc = Tbc_.clone();
    Tcb = cv::Mat::eye(4,4,CV_32F);
    Tcb.rowRange(0,3).colRange(0,3) = Tbc.rowRange(0,3).colRange(0,3).t();
    Tcb.rowRange(0,3).col(3) = -Tbc.rowRange(0,3).colRange(0,3).t()*Tbc.rowRange(0,3).col(3);
    Cov = cv::Mat::eye(6,6,CV_32F);
    const float ng2 = ng*ng;
    const float na2 = na*na;
    Cov.at<float>(0,0) = ng2;
    Cov.at<float>(1,1) = ng2;
    Cov.at<float>(2,2) = ng2;
    Cov.at<float>(3,3) = na2;
    Cov.at<float>(4,4) = na2;
    Cov.at<float>(5,5) = na2;
    CovWalk = cv::Mat::eye(6,6,CV_32F);
    const float ngw2 = ngw*ngw;
    const float naw2 = naw*naw;
    CovWalk.at<float>(0,0) = ngw2;
    CovWalk.at<float>(1,1) = ngw2;
    CovWalk.at<float>(2,2) = ngw2;
    CovWalk.at<float>(3,3) = naw2;
    CovWalk.at<float>(4,4) = naw2;
    CovWalk.at<float>(5,5) = naw2;
    }
    
public:
    cv::Mat Tcb;  //b to camera
    cv::Mat Tbc;
    cv::Mat Cov, CovWalk; // imu协方差， 随机游走协方差
};

//Integration of 1 gyro measurement
class IntegratedRotation
{
public:
    IntegratedRotation(){}
    IntegratedRotation(const cv::Point3f &angVel, const Bias &imuBias, const float &time);

public:
    float deltaT; //integration time
    cv::Mat deltaR; //integrated rotation
    cv::Mat rightJ; // right jacobian
};



} // namespace lvio_fusion
#endif // lvio_fusion_IMU_H
