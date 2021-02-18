#include "lvio_fusion/estimator.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/manager.h"

#include <opencv2/core/eigen.hpp>
#include <sys/sysinfo.h>

double epsilon = 1e-3;
int num_threads = std::min(8, std::max(1, (int)(0.75 * get_nprocs())));

namespace lvio_fusion
{

Estimator::Estimator(std::string &config_path)
    : config_file_path_(config_path) {}

bool Estimator::Init(int use_imu, int use_lidar, int use_navsat, int use_loop, int use_adapt)
{
    LOG(INFO) << "System info:\n\tepsilon: " << epsilon << "\n\tnum_threads: " << num_threads;

    // read from config file
    if (!Config::SetParameterFile(config_file_path_))
    {
        return false;
    }

    // read camera intrinsics and extrinsics
    bool undistort = Config::Get<int>("undistort");
    cv::Mat cv_cam0_to_body = Config::Get<cv::Mat>("cam0_to_body");
    cv::Mat cv_cam1_to_body = Config::Get<cv::Mat>("cam1_to_body");
    Matrix4d cam0_to_body, cam1_to_body;
    cv::cv2eigen(cv_cam0_to_body, cam0_to_body);
    cv::cv2eigen(cv_cam1_to_body, cam1_to_body);
    // first camera
    Matrix3d R_cam0_to_body(cam0_to_body.block(0, 0, 3, 3));
    Quaterniond q_cam0_to_body(R_cam0_to_body);
    Vector3d t_cam0_to_body(0, 0, 0);
    t_cam0_to_body << cam0_to_body(0, 3), cam0_to_body(1, 3), cam0_to_body(2, 3);
    if (undistort)
    {
        Camera::Create(Config::Get<double>("camera0.fx"),
                       Config::Get<double>("camera0.fy"),
                       Config::Get<double>("camera0.cx"),
                       Config::Get<double>("camera0.cy"),
                       Config::Get<double>("camera0.k1"),
                       Config::Get<double>("camera0.k2"),
                       Config::Get<double>("camera0.p1"),
                       Config::Get<double>("camera0.p2"),
                       SE3d(q_cam0_to_body, t_cam0_to_body));
    }
    else
    {
        Camera::Create(Config::Get<double>("camera0.fx"),
                       Config::Get<double>("camera0.fy"),
                       Config::Get<double>("camera0.cx"),
                       Config::Get<double>("camera0.cy"),
                       SE3d(q_cam0_to_body, t_cam0_to_body));
    }
    // second camera
    Matrix3d R_cam1_to_body(cam1_to_body.block(0, 0, 3, 3));
    Quaterniond q_cam1_to_body(R_cam1_to_body);
    Vector3d t_cam1_to_body(0, 0, 0);
    t_cam1_to_body << cam1_to_body(0, 3), cam1_to_body(1, 3), cam1_to_body(2, 3);
    if (undistort)
    {
        Camera::Create(Config::Get<double>("camera1.fx"),
                       Config::Get<double>("camera1.fy"),
                       Config::Get<double>("camera1.cx"),
                       Config::Get<double>("camera1.cy"),
                       Config::Get<double>("camera1.k1"),
                       Config::Get<double>("camera1.k2"),
                       Config::Get<double>("camera1.p1"),
                       Config::Get<double>("camera1.p2"),
                       SE3d(q_cam1_to_body, t_cam1_to_body));
    }
    else
    {
        Camera::Create(Config::Get<double>("camera1.fx"),
                       Config::Get<double>("camera1.fy"),
                       Config::Get<double>("camera1.cx"),
                       Config::Get<double>("camera1.cy"),
                       SE3d(q_cam1_to_body, t_cam1_to_body));
    }

    // create components and links
    frontend = Frontend::Ptr(new Frontend(
        Config::Get<int>("num_features"),
        Config::Get<int>("num_features_init"),
        Config::Get<int>("num_features_tracking"),
        Config::Get<int>("num_features_tracking_bad"),
        Config::Get<int>("num_features_needed_for_keyframe")));

    backend = Backend::Ptr(new Backend(
        Config::Get<double>("windows_size"),
        use_adapt));

    frontend->SetBackend(backend);
    backend->SetFrontend(frontend);

    PoseGraph::Instance().SetFrontend(frontend);

    if (use_loop)
    {
        relocator = Relocator::Ptr(new Relocator(
            Config::Get<int>("relocator_mode"),
            Config::Get<int>("threshold")));
        relocator->SetBackend(backend);
    }

    if (use_navsat)
    {
        Navsat::Create();
    }

    if (use_imu)
    {
        Imu::Create(SE3d());
        initializer = Initializer::Ptr(new Initializer);
        backend->SetInitializer(initializer);
    }

    if (use_lidar)
    {
        cv::Mat cv_lidar_to_body = Config::Get<cv::Mat>("lidar_to_body");
        Matrix4d lidar_to_body;
        cv::cv2eigen(cv_lidar_to_body, lidar_to_body);
        Matrix3d R_lidar_to_body(lidar_to_body.block(0, 0, 3, 3));
        Quaterniond q_lidar_to_body(R_lidar_to_body);
        Vector3d t_lidar_to_body(0, 0, 0);
        t_lidar_to_body << lidar_to_body(0, 3), lidar_to_body(1, 3), lidar_to_body(2, 3);
        Lidar::Create(Config::Get<double>("resolution"),
                      SE3d(q_lidar_to_body, t_lidar_to_body));

        association = FeatureAssociation::Ptr(new FeatureAssociation(
            Config::Get<int>("num_scans"),
            Config::Get<int>("horizon_scan"),
            Config::Get<double>("ang_res_y"),
            Config::Get<double>("ang_bottom"),
            Config::Get<int>("ground_rows"),
            Config::Get<double>("cycle_time"),
            Config::Get<double>("min_range"),
            Config::Get<double>("max_range"),
            Config::Get<int>("deskew"),
            Config::Get<int>("spacing")));

        mapping = Mapping::Ptr(new Mapping);
        mapping->SetFeatureAssociation(association);

        backend->SetMapping(mapping);

        if (relocator)
        {
            relocator->SetMapping(mapping);
        }
    }
    return true;
}

void Estimator::InputImage(double time, cv::Mat &left_image, cv::Mat &right_image, std::vector<DetectedObject> objects)
{
    Frame::Ptr new_frame = Frame::Create();
    new_frame->time = time;
    cv::undistort(left_image, new_frame->image_left, Camera::Get(0)->K, Camera::Get(0)->D);
    cv::undistort(right_image, new_frame->image_right, Camera::Get(1)->K, Camera::Get(1)->D);
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
