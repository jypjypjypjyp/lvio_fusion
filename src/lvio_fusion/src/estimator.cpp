#include <fstream>

#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/pose_only_reprojection_error.hpp"
#include "lvio_fusion/ceres/two_frame_reprojection_error.hpp"
#include "lvio_fusion/ceres/vehicle_motion_error.hpp"
#include "lvio_fusion/config.h"
#include "lvio_fusion/estimator.h"
#include "lvio_fusion/frame.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Matrix2d TwoFrameReprojectionError::sqrt_info = Matrix2d::Identity();
Matrix2d PoseOnlyReprojectionError::sqrt_info = Matrix2d::Identity();
Matrix3d NavsatError::sqrt_info = 10 * Matrix3d::Identity();

Estimator::Estimator(std::string &config_path)
    : config_file_path_(config_path) {}

bool Estimator::Init()
{
    // read from config file
    if (!Config::SetParameterFile(config_file_path_))
    {
        return false;
    }

    // read camera intrinsics and extrinsics
    cv::Mat cv_body_T_cam0 = Config::Get<cv::Mat>("body_T_cam0");
    cv::Mat cv_body_T_cam1 = Config::Get<cv::Mat>("body_T_cam1");
    Matrix4d body_T_cam0, body_T_cam1;
    cv::cv2eigen(cv_body_T_cam0, body_T_cam0);
    cv::cv2eigen(cv_body_T_cam1, body_T_cam1);
    // first camera
    Matrix3d R_body_T_cam0(body_T_cam0.block(0, 0, 3, 3));
    Quaterniond q_body_T_cam0(R_body_T_cam0);
    Vector3d t_body_T_cam0(0, 0, 0);
    t_body_T_cam0 << body_T_cam0(0, 3), body_T_cam0(1, 3), body_T_cam0(2, 3);
    Camera::Ptr camera1(new Camera(Config::Get<double>("camera1.fx"),
                                   Config::Get<double>("camera1.fy"),
                                   Config::Get<double>("camera1.cx"),
                                   Config::Get<double>("camera1.cy"),
                                   SE3d(SO3d(q_body_T_cam0), t_body_T_cam0)));
    LOG(INFO) << "Camera 1"
              << " extrinsics: " << t_body_T_cam0.transpose();
    // second camera
    Matrix3d R_body_T_cam1(body_T_cam1.block(0, 0, 3, 3));
    Quaterniond q_body_T_cam1(R_body_T_cam1);
    Vector3d t_body_T_cam1(0, 0, 0);
    t_body_T_cam1 << body_T_cam1(0, 3), body_T_cam1(1, 3), body_T_cam1(2, 3);
    Camera::Ptr camera2(new Camera(Config::Get<double>("camera2.fx"),
                                   Config::Get<double>("camera2.fy"),
                                   Config::Get<double>("camera2.cx"),
                                   Config::Get<double>("camera2.cy"),
                                   SE3d(SO3d(q_body_T_cam1), t_body_T_cam1)));
    LOG(INFO) << "Camera 2"
              << " extrinsics: " << t_body_T_cam1.transpose();

    // create components and links
    frontend = Frontend::Ptr(new Frontend(
        Config::Get<int>("num_features"),
        Config::Get<int>("num_features_init"),
        Config::Get<int>("num_features_tracking"),
        Config::Get<int>("num_features_tracking_bad"),
        Config::Get<int>("num_features_needed_for_keyframe")
    ));
    backend = Backend::Ptr(new Backend(
        Config::Get<double>("range")
    ));
    map = Map::Ptr(new Map());
    auto navsat_map = NavsatMap::Ptr(new NavsatMap(map));
    map->navsat_map = navsat_map;

    frontend->SetBackend(backend);
    frontend->SetMap(map);
    frontend->SetCameras(camera1, camera2);
    frontend->flags += Flag::Stereo;

    backend->SetMap(map);
    backend->SetCameras(camera1, camera2);
    backend->SetFrontend(frontend);

    // semantic map
    if (Config::Get<int>("is_semantic"))
    {
        frontend->flags += Flag::Semantic;
    }

    return true;
}

void Estimator::InputImage(double time, cv::Mat &left_image, cv::Mat &right_image, std::vector<DetectedObject> objects)
{
    Frame::Ptr new_frame = Frame::Create();
    new_frame->time = time;
    new_frame->left_image = left_image;
    new_frame->right_image = right_image;
    new_frame->objects = objects;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO status:" << (success ? "success" : "failed") << ",VO cost time: " << time_used.count() << " seconds.";
}

void Estimator::InputPointCloud(double time, PointCloudI::Ptr point_cloud)
{
    
}

void Estimator::InputIMU(double time, Vector3d acc, Vector3d gyr)
{
}

void Estimator::InputNavSat(double time, double x, double y, double z, double posAccuracy)
{
    NavsatPoint new_point(time, x, y, z);
    map->navsat_map->AddPoint(new_point);
}

} // namespace lvio_fusion
