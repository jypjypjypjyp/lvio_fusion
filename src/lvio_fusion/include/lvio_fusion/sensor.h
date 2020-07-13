#ifndef lvio_fusion_SENSOR_H
#define lvio_fusion_SENSOR_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

template <typename T>
class Sensor
{
public:
    typedef std::shared_ptr<Sensor<T>> Ptr;

    Sensor(SE3d extrinsic) : extrinsic(extrinsic) {}

    // coordinate transform: world, sensor, pixel
    virtual Matrix<T, 3, 1> World2Sensor(const Matrix<T, 3, 1> &p_w, const Sophus::SE3<T> &T_c_w)
    {
        throw NotImplemented();
    }

    virtual Matrix<T, 3, 1> Sensor2World(const Matrix<T, 3, 1> &p_c, const Sophus::SE3<T> &T_c_w)
    {
        throw NotImplemented();
    }

    virtual Matrix<T, 2, 1> Sensor2Pixel(const Matrix<T, 3, 1> &p_c)
    {
        throw NotImplemented();
    }

    virtual Matrix<T, 3, 1> Pixel2Sensor(const Matrix<T, 2, 1> &p_p, T depth = 1)
    {
        throw NotImplemented();
    }

    virtual Matrix<T, 3, 1> Pixel2World(const Matrix<T, 2, 1> &p_p, const Sophus::SE3<T> &T_c_w, T depth = 1)
    {
        return Sensor2World(Pixel2Sensor(p_p, depth), T_c_w);
    }

    virtual Matrix<T, 2, 1> World2Pixel(const Matrix<T, 3, 1> &p_w, const Sophus::SE3<T> &T_c_w)
    {
        return Sensor2Pixel(World2Sensor(p_w, T_c_w));
    }

    SE3d extrinsic;
};

typedef Sensor<double> Sensord;

} // namespace lvio_fusion
#endif // lvio_fusion_SENSOR_H
