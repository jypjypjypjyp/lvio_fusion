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
                       SE3d(q_base_to_cam0, t_base_to_cam0));
    }
    else
    {
        Camera::Create(Config::Get<double>("camera0.fx"),
                       Config::Get<double>("camera0.fy"),
                       Config::Get<double>("camera0.cx"),
                       Config::Get<double>("camera0.cy"),
                       SE3d(q_base_to_cam0, t_base_to_cam0));
    }
    // second camera
    Matrix3d R_base_to_cam1(base_to_cam1.block(0, 0, 3, 3));
    Quaterniond q_base_to_cam1(R_base_to_cam1);
    Vector3d t_base_to_cam1(0, 0, 0);
    t_base_to_cam1 << base_to_cam1(0, 3), base_to_cam1(1, 3), base_to_cam1(2, 3);
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
                       SE3d(q_base_to_cam1, t_base_to_cam1));
    }
    else
    {
        Camera::Create(Config::Get<double>("camera1.fx"),
                       Config::Get<double>("camera1.fy"),
                       Config::Get<double>("camera1.cx"),
                       Config::Get<double>("camera1.cy"),
                       SE3d(q_base_to_cam1, t_base_to_cam1));
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
