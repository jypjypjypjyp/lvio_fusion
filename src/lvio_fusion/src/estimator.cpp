#include "lvio_fusion/estimator.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/config.h"
#include "lvio_fusion/frame.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Matrix2d TwoFrameReprojectionError::sqrt_info = Matrix2d::Identity();
Matrix2d PoseOnlyReprojectionError::sqrt_info = Matrix2d::Identity();
Matrix3d LidarEdgeError::sqrt_info = 0.1 * Matrix3d::Identity();
double LidarPlaneError::sqrt_info = 0.1;
Matrix3d NavsatError::sqrt_info = 10 * Matrix3d::Identity();

Estimator::Estimator(std::string &config_path)
    : config_file_path_(config_path) {}

bool Estimator::Init(int use_imu, int use_lidar, int use_navsat, int use_loop, int is_semantic)
{
    // read from config file
    if (!Config::SetParameterFile(config_file_path_))
    {
        return false;
    }

    // read camera intrinsics and extrinsics
    cv::Mat cv_base_to_cam0 = Config::Get<cv::Mat>("base_to_cam0");
    cv::Mat cv_base_to_cam1 = Config::Get<cv::Mat>("base_to_cam1");
    Matrix4d base_to_cam0, base_to_cam1;
    cv::cv2eigen(cv_base_to_cam0, base_to_cam0);
    cv::cv2eigen(cv_base_to_cam1, base_to_cam1);
    // first camera
    Matrix3d R_base_to_cam0(base_to_cam0.block(0, 0, 3, 3));
    Quaterniond q_base_to_cam0(R_base_to_cam0);
    Vector3d t_base_to_cam0(0, 0, 0);
    t_base_to_cam0 << base_to_cam0(0, 3), base_to_cam0(1, 3), base_to_cam0(2, 3);
    Camera::Ptr camera1(new Camera(Config::Get<double>("camera1.fx"),
                                   Config::Get<double>("camera1.fy"),
                                   Config::Get<double>("camera1.cx"),
                                   Config::Get<double>("camera1.cy"),
                                   SE3d(q_base_to_cam0, t_base_to_cam0)));
    LOG(INFO) << "Camera 1"
              << " extrinsics: " << t_base_to_cam0.transpose();
    // second camera
    Matrix3d R_base_to_cam1(base_to_cam1.block(0, 0, 3, 3));
    Quaterniond q_base_to_cam1(R_base_to_cam1);
    Vector3d t_base_to_cam1(0, 0, 0);
    t_base_to_cam1 << base_to_cam1(0, 3), base_to_cam1(1, 3), base_to_cam1(2, 3);
    Camera::Ptr camera2(new Camera(Config::Get<double>("camera2.fx"),
                                   Config::Get<double>("camera2.fy"),
                                   Config::Get<double>("camera2.cx"),
                                   Config::Get<double>("camera2.cy"),
                                   SE3d(q_base_to_cam1, t_base_to_cam1)));
    LOG(INFO) << "Camera 2"
              << " extrinsics: " << t_base_to_cam1.transpose();

    // create components and links
    frontend = Frontend::Ptr(new Frontend(
        Config::Get<int>("num_features"),
        Config::Get<int>("num_features_init"),
        Config::Get<int>("num_features_tracking"),
        Config::Get<int>("num_features_tracking_bad"),
        Config::Get<int>("num_features_needed_for_keyframe")
        ));


    backend = Backend::Ptr(new Backend(
        Config::Get<double>("range")));

    map = Map::Ptr(new Map());

    frontend->SetBackend(backend);
    frontend->SetMap(map);
    frontend->SetCameras(camera1, camera2);
    frontend->flags += Flag::Stereo;

    backend->SetMap(map);
    backend->SetCameras(camera1, camera2);
    backend->SetFrontend(frontend);

    mapping = Mapping::Ptr(new Mapping());
    mapping->SetCamera(camera1);
    mapping->SetMap(map);
    mapping->SetBackend(backend);

    if (use_loop)
    {
        relocation = Relocation::Ptr(new Relocation(Config::Get<std::string>("voc_path")));
        relocation->SetCameras(camera1, camera2);
        relocation->SetMap(map);
        frontend->SetRelocation(relocation);
    }
    if (use_navsat)
    {
        NavsatMap::Ptr navsat_map(new NavsatMap(map));
        map->navsat_map = navsat_map;
        frontend->flags += Flag::GNSS;
    }

    if (use_imu)
    {
        Imu::Ptr imu(new Imu(SE3d()));

        initializer = Initializer::Ptr(new Initializer);
        //NEWADD
        initializer->SetMap(map);
        initializer->SetFrontend(frontend);
       
        //NEWADDEND
        backend->SetImu(imu);
        backend->SetInitializer(initializer);

        frontend->SetImu(imu);
        //NEWADD
        frontend->SetInitializer(initializer);
        //NEWADDEND
        frontend->flags += Flag::IMU;
    }

    if (use_lidar)
    {
        cv::Mat cv_base_to_lidar = Config::Get<cv::Mat>("base_to_lidar");
        Matrix4d base_to_lidar;
        cv::cv2eigen(cv_base_to_lidar, base_to_lidar);
        Matrix3d R_base_to_lidar(base_to_lidar.block(0, 0, 3, 3));
        Quaterniond q_base_to_lidar(R_base_to_lidar);
        Vector3d t_base_to_lidar(0, 0, 0);
        t_base_to_lidar << base_to_lidar(0, 3), base_to_lidar(1, 3), base_to_lidar(2, 3);
        Lidar::Ptr lidar(new Lidar(SE3d(q_base_to_lidar, t_base_to_lidar)));

        scan_registration = ScanRegistration::Ptr(new ScanRegistration(
            Config::Get<int>("num_scans"),
            Config::Get<double>("cycle_time"),
            Config::Get<double>("min_range"),
            Config::Get<double>("max_range"),
            Config::Get<int>("deskew")));
        scan_registration->SetLidar(lidar);
        scan_registration->SetMap(map);

        mapping->SetLidar(lidar);

        backend->SetLidar(lidar);
        backend->SetScanRegistration(scan_registration);
        if(relocation)
        {
            relocation->SetLidar(lidar);
            relocation->SetScanRegistration(scan_registration);
        }

        frontend->flags += Flag::Laser;
    }

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
    new_frame->image_left = left_image;
    new_frame->image_right = right_image;
    new_frame->objects = objects;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO status:" << (success ? "success" : "failed") << ",VO cost time: " << time_used.count() << " seconds.";
}

void Estimator::InputPointCloud(double time, Point3Cloud::Ptr point_cloud)
{
    auto t1 = std::chrono::steady_clock::now();
    scan_registration->AddScan(time, point_cloud);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    if (time_used.count() > 1e-2)
        LOG(INFO) << "Scan Registration cost time: " << time_used.count() << " seconds.";
}

void Estimator::InputIMU(double time, Vector3d acc, Vector3d gyr)
{
    frontend->AddImu(time, acc, gyr);
}

void Estimator::InputNavSat(double time, double x, double y, double z, double posAccuracy)
{
    NavsatPoint new_point(time, x, y, z);
    map->navsat_map->AddPoint(new_point);
}

} // namespace lvio_fusion
