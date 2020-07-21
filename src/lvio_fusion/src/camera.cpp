#include "lvio_fusion/camera.h"

namespace lvio_fusion
{

Camerad::Camerad()
{
}

Vector3d Camerad::World2Sensor(const Vector3d &p_w, const SE3d &T_c_w)
{
    return extrinsic * T_c_w * p_w;
}

Vector3d Camerad::Sensor2World(const Vector3d &p_c, const SE3d &T_c_w)
{
    return T_c_w.inverse() * extrinsic.inverse() * p_c;
}

Vector2d Camerad::Sensor2Pixel(const Vector3d &p_c)
{
    return Vector2d(
        fx * p_c(0, 0) / p_c(2, 0) + cx,
        fy * p_c(1, 0) / p_c(2, 0) + cy);
}

Vector3d Camerad::Pixel2Sensor(const Vector2d &p_p, double depth)
{
    return Vector3d(
        (p_p(0, 0) - cx) * depth / fx,
        (p_p(1, 0) - cy) * depth / fy,
        depth);
}

Vector2d Camerad::World2Pixel(const Vector3d &p_w, const SE3d &T_c_w)
{
    return Sensor2Pixel(World2Sensor(p_w, T_c_w));
}

Vector3d Camerad::Pixel2World(const Vector2d &p_p, const SE3d &T_c_w, double depth)
{
    return Sensor2World(Pixel2Sensor(p_p, depth), T_c_w);
}

} // namespace lvio_fusion
