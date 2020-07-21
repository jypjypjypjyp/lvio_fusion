

#ifndef lvio_fusion_CAMERA_H
#define lvio_fusion_CAMERA_H

#include "lvio_fusion/common.h"

namespace lvio_fusion {

// Pinhole stereo camera model
class Camerad {
   public:
    typedef std::shared_ptr<Camerad> Ptr;

    Camerad();

    Camerad(double fx, double fy, double cx, double cy,const SE3d &pose)
        : fx(fx), fy(fy), cx(cx), cy(cy), extrinsic(pose) {}

    // return intrinsic matrix
    Matrix3d K() const {
        Matrix3d k;
        k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return k;
    }

    // coordinate transform: world, camera, pixel
    Vector3d World2Sensor(const Vector3d &p_w, const SE3d &T_c_w);

    Vector3d Sensor2World(const Vector3d &p_c, const SE3d &T_c_w);

    Vector2d Sensor2Pixel(const Vector3d &p_c);

    Vector3d Pixel2Sensor(const Vector2d &p_p, double depth = 1);

    Vector3d Pixel2World(const Vector2d &p_p, const SE3d &T_c_w, double depth = 1);

    Vector2d World2Pixel(const Vector3d &p_w, const SE3d &T_c_w);

    double fx = 0, fy = 0, cx = 0, cy = 0;  // Camera intrinsics
    SE3d extrinsic;                               // extrinsic, from stereo camera to single camera
};

}  // namespace lvio_fusion
#endif  // lvio_fusion_CAMERA_H
