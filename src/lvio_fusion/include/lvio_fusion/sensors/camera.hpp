

// #ifndef lvio_fusion_CAMERA_H
// #define lvio_fusion_CAMERA_H

// #include "lvio_fusion/common.h"
// #include "lvio_fusion/sensor.h"

// namespace lvio_fusion
// {

// // Pinhole stereo camera model
// // template <typename T>
// class Camerad //: public Sensor<T>
// {
// public:
//     // typedef std::shared_ptr<Camera<T>> Ptr;

//     // Camera(double fx, double fy, double cx, double cy, const SE3d &extrinsic)
//     //     : fx(fx), fy(fy), cx(cx), cy(cy), Sensor<T>(extrinsic) {}

//     // // return intrinsic matrix
//     // Matrix3d K() const
//     // {
//     //     Matrix3d k;
//     //     k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
//     //     return k;
//     // }

//     // // coordinate transform: world, sensor, pixel
//     // virtual Matrix<T, 3, 1> World2Sensor(const Matrix<T, 3, 1> &p_w, const Sophus::SE3<T> &T_c_w)
//     // {
//     //     return Sensor<T>::extrinsic_t * T_c_w * p_w;
//     // }

//     // virtual Matrix<T, 3, 1> Sensor2World(const Matrix<T, 3, 1> &p_c, const Sophus::SE3<T> &T_c_w)
//     // {
//     //     return T_c_w.inverse() * Sensor<T>::extrinsic_t.inverse() * p_c;
//     // }

//     // virtual Matrix<T, 2, 1> Sensor2Pixel(const Matrix<T, 3, 1> &p_c)
//     // {
//     //     return Matrix<T, 2, 1>(
//     //         T(fx) * p_c(0, 0) / p_c(2, 0) + T(cx),
//     //         T(fy) * p_c(1, 0) / p_c(2, 0) + T(cy));
//     // }

//     // virtual Matrix<T, 3, 1> Pixel2Sensor(const Matrix<T, 2, 1> &p_p, T depth = T(1))
//     // {
//     //     return Matrix<T, 3, 1>(
//     //         (p_p(0, 0) - T(cx)) * depth / T(fx),
//     //         (p_p(1, 0) - T(cy)) * depth / T(fy),
//     //         depth);
//     // }

//     // template <typename NEW_TYPE>
//     // Camera<NEW_TYPE> cast()
//     // {
//     //     return Camera<NEW_TYPE>(fx, fy, cx, cy, Sensor<T>::extrinsic);
//     // }

//     typedef std::shared_ptr<Camerad> Ptr;
//     Camerad(double fx, double fy, double cx, double cy, const SE3d &extrinsic)
//         : fx(fx), fy(fy), cx(cx), cy(cy), extrinsic(extrinsic) {}

//     // return intrinsic matrix
//     Matrix3d K() const
//     {
//         Matrix3d k;
//         k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
//         return k;
//     }

//     Vector3d World2Sensor(const Vector3d &p_w, const SE3d &T_c_w)
//     {
//         return extrinsic * T_c_w * p_w;
//     }
//     Vector3d Sensor2World(const Vector3d &p_c, const SE3d &T_c_w)
//     {
//         return T_c_w.inverse() * extrinsic.inverse() * p_c;
//     }
//     Vector2d Sensor2Pixel(const Vector3d &p_c)
//     {
//         return Vector2d(
//             fx * p_c(0, 0) / p_c(2, 0) + cx,
//             fy * p_c(1, 0) / p_c(2, 0) + cy);
//     }
//     Vector3d Pixel2Sensor(const Vector2d &p_p, double depth = 1)
//     {
//         return Vector3d(
//             (p_p(0, 0) - cx) * depth / fx,
//             (p_p(1, 0) - cy) * depth / fy,
//             depth);
//     }
//     Vector2d World2Pixel(const Vector3d &p_w, const SE3d &T_c_w)
//     {
//         return Sensor2Pixel(World2Sensor(p_w, T_c_w));
//     }
//     Vector3d Pixel2World(const Vector2d &p_p, const SE3d &T_c_w, double depth = 1)
//     {
//         return Sensor2World(Pixel2Sensor(p_p, depth), T_c_w);
//     }

//     double fx = 0, fy = 0, cx = 0, cy = 0; // Camera intrinsics
//     SE3d extrinsic;
// };

// // typedef Camera<double> Camerad;

// } // namespace lvio_fusion
// #endif // lvio_fusion_CAMERA_H
