#include "lvio_fusion/estimator.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/manager.h"

#include <opencv2/core/eigen.hpp>

double epsilon = 1e-3;

namespace lvio_fusion
{

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
    Camera::Create(Config::Get<double>("camera1.fx"),
                   Config::Get<double>("camera1.fy"),
                   Config::Get<double>("camera1.cx"),
                   Config::Get<double>("camera1.cy"),
                   SE3d(q_base_to_cam0, t_base_to_cam0));
    LOG(INFO) << "Camera 1"
              << " extrinsics: " << t_base_to_cam0.transpose();
    // second camera
    Matrix3d R_base_to_cam1(base_to_cam1.block(0, 0, 3, 3));
    Quaterniond q_base_to_cam1(R_base_to_cam1);
    Vector3d t_base_to_cam1(0, 0, 0);
    t_base_to_cam1 << base_to_cam1(0, 3), base_to_cam1(1, 3), base_to_cam1(2, 3);
    Camera::Create(Config::Get<double>("camera2.fx"),
                   Config::Get<double>("camera2.fy"),
                   Config::Get<double>("camera2.cx"),
                   Config::Get<double>("camera2.cy"),
                   SE3d(q_base_to_cam1, t_base_to_cam1));
    LOG(INFO) << "Camera 2"
              << " extrinsics: " << t_base_to_cam1.transpose();

    // create components and links
    frontend = Frontend::Ptr(new Frontend(
        Config::Get<int>("num_features"),
        Config::Get<int>("num_features_init"),
        Config::Get<int>("num_features_tracking"),
        Config::Get<int>("num_features_tracking_bad"),
        Config::Get<int>("num_features_needed_for_keyframe")));

    backend = Backend::Ptr(new Backend(
        Config::Get<double>("delay")));

    frontend->SetBackend(backend);
    flags += Flag::Stereo;

    backend->SetFrontend(frontend);

    pose_graph = PoseGraph::Ptr(new PoseGraph);
    pose_graph->SetFrontend(frontend);

    if (use_loop)
    {
        relocator = LoopDetector::Ptr(new LoopDetector(
            Config::Get<std::string>("voc_path")));
        relocator->SetFrontend(frontend);
        relocator->SetBackend(backend);
        relocator->SetPoseGraph(pose_graph);
    }

    if (use_navsat)
    {
        Navsat::Create();
        Navsat::Get()->SetPoseGraph(pose_graph);
        flags += Flag::GNSS;
    }

    if (use_imu)
    {
        Imu::Create(SE3d());
        initializer = Initializer::Ptr(new Initializer);
        backend->SetInitializer(initializer);
        flags += Flag::IMU;
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
        Lidar::Create(Config::Get<double>("resolution"),
                      SE3d(q_base_to_lidar, t_base_to_lidar));

        association = FeatureAssociation::Ptr(new FeatureAssociation(
            Config::Get<int>("num_scans"),
            Config::Get<int>("horizon_scan"),
            Config::Get<double>("ang_res_y"),
            Config::Get<double>("ang_bottom"),
            Config::Get<int>("ground_rows"),
            Config::Get<double>("cycle_time"),
            Config::Get<double>("min_range"),
            Config::Get<double>("max_range"),
            Config::Get<int>("deskew")));

        mapping = Mapping::Ptr(new Mapping);
        mapping->SetFeatureAssociation(association);

        backend->SetMapping(mapping);

        if (relocator)
        {
            relocator->SetFeatureAssociation(association);
            relocator->SetMapping(mapping);
        }

        flags += Flag::Laser;
    }

    // semantic map
    if (is_semantic)
    {
        flags += Flag::Semantic;
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
    association->AddScan(time, point_cloud);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    if (time_used.count() > 1e-2)
        LOG(INFO) << "Lidar Preprocessing cost time: " << time_used.count() << " seconds.";
}

void Estimator::InputIMU(double time, Vector3d acc, Vector3d gyr)
{
    frontend->AddImu(time, acc, gyr);
}

void Estimator::InputNavSat(double time, double x, double y, double z, double posAccuracy)
{
    Navsat::Get()->AddPoint(time, x, y, z);
}

} // namespace lvio_fusion
