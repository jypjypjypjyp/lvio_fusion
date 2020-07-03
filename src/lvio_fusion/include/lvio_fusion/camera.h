

#ifndef lvio_fusion_CAMERA_H
#define lvio_fusion_CAMERA_H

#include "lvio_fusion/common.h"

namespace lvio_fusion {

// Pinhole stereo camera model
class Camera {
   public:
    typedef std::shared_ptr<Camera> Ptr;

    Camera();

    Camera(double fx, double fy, double cx, double cy,const SE3 &pose)
        : fx(fx), fy(fy), cx(cx), cy(cy), pose(pose) {}

    // return intrinsic matrix
    Matrix3d K() const {
        Matrix3d k;
        k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return k;
    }

    // coordinate transform: world, camera, pixel
    Vector3d world2camera(const Vector3d &p_w, const SE3 &T_c_w);

    Vector3d camera2world(const Vector3d &p_c, const SE3 &T_c_w);

    Vector2d camera2pixel(const Vector3d &p_c);

    Vector3d pixel2camera(const Vector2d &p_p, double depth = 1);

    Vector3d pixel2world(const Vector2d &p_p, const SE3 &T_c_w, double depth = 1);

    Vector2d world2pixel(const Vector3d &p_w, const SE3 &T_c_w);

    double fx = 0, fy = 0, cx = 0, cy = 0;  // Camera intrinsics
    SE3 pose;                               // extrinsic, from stereo camera to single camera
};

}  // namespace lvio_fusion
#endif  // lvio_fusion_CAMERA_H
