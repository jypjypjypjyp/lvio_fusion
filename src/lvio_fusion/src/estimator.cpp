#include "lvio_fusion/estimator.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/manager.h"

#include <opencv2/core/eigen.hpp>
#include <sys/sysinfo.h>

const double epsilon = 1e-3;
const int num_threads = std::min(8, std::max(1, (int)(0.75 * get_nprocs())));
const double max_speed = 40;

namespace lvio_fusion
{

Estimator::Estimator(std::string &config_path) : config_file_path_(config_path) {}

bool Estimator::Init(int use_imu, int use_lidar, int use_navsat, int use_loop, int use_adapt, int use_navigation)//NAVI
{
    LOG(INFO) << "System info:\n\tepsilon: " << epsilon << "\n\tnum_threads: " << num_threads;

    // read from config file
    if (!Config::SetParameterFile(config_file_path_))
    {
        return false;
    }

    // read camera intrinsics and extrinsics
    bool undistort = Config::Get<int>("undistort");
    cv::Mat cv_body_to_cam0 = Config::Get<cv::Mat>("body_to_cam0");
    cv::Mat cv_body_to_cam1 = Config::Get<cv::Mat>("body_to_cam1");
    Matrix4d body_to_cam0, body_to_cam1;
    cv::cv2eigen(cv_body_to_cam0, body_to_cam0);
    cv::cv2eigen(cv_body_to_cam1, body_to_cam1);
    // first camera
    Matrix3d R_body_to_cam0(body_to_cam0.block(0, 0, 3, 3));
    Quaterniond q_body_to_cam0(R_body_to_cam0);
    Vector3d t_body_to_cam0(0, 0, 0);
    t_body_to_cam0 << body_to_cam0(0, 3), body_to_cam0(1, 3), body_to_cam0(2, 3);
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
                       SE3d(q_body_to_cam0, t_body_to_cam0));
    }
    else
    {
        Camera::Create(Config::Get<double>("camera0.fx"),
                       Config::Get<double>("camera0.fy"),
                       Config::Get<double>("camera0.cx"),
                       Config::Get<double>("camera0.cy"),
                       SE3d(q_body_to_cam0, t_body_to_cam0));
    }
    // second camera
    Matrix3d R_body_to_cam1(body_to_cam1.block(0, 0, 3, 3));
    Quaterniond q_body_to_cam1(R_body_to_cam1);
    Vector3d t_body_to_cam1(0, 0, 0);
    t_body_to_cam1 << body_to_cam1(0, 3), body_to_cam1(1, 3), body_to_cam1(2, 3);
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
                       SE3d(q_body_to_cam1, t_body_to_cam1));
    }
    else
    {
        Camera::Create(Config::Get<double>("camera1.fx"),
                       Config::Get<double>("camera1.fy"),
                       Config::Get<double>("camera1.cx"),
                       Config::Get<double>("camera1.cy"),
                       SE3d(q_body_to_cam1, t_body_to_cam1));
    }
    Camera::baseline = (t_body_to_cam0 - t_body_to_cam1).norm();

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
        Navsat::Create(Config::Get<double>("accuracy"));
    }

    if (use_imu)
    {
        initializer = Initializer::Ptr(new Initializer);
        backend->SetInitializer(initializer);

        double acc_n = Config::Get<double>("acc_n");
        double gyr_n = Config::Get<double>("gyr_n");
        double acc_w = Config::Get<double>("acc_w");
        double gyr_w = Config::Get<double>("gyr_w");
        double g_norm = Config::Get<double>("g_norm");
        Imu::Create(SE3d(), acc_n, acc_w, gyr_n, gyr_w, g_norm);
    }

    if (use_lidar)
    {
        cv::Mat cv_body_to_lidar = Config::Get<cv::Mat>("body_to_lidar");
        Matrix4d body_to_lidar;
        cv::cv2eigen(cv_body_to_lidar, body_to_lidar);
        Matrix3d R_body_to_lidar(body_to_lidar.block(0, 0, 3, 3));
        Quaterniond q_body_to_lidar(R_body_to_lidar);
        Vector3d t_body_to_lidar(0, 0, 0);
        t_body_to_lidar << body_to_lidar(0, 3), body_to_lidar(1, 3), body_to_lidar(2, 3);
        Lidar::Create(Config::Get<double>("resolution"),
                      SE3d(q_body_to_lidar, t_body_to_lidar));

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
    else{    //NAVI
        use_navigation=0;
    }
    //NAVI
    if(use_navigation)
    {
        gridmap = Gridmap::Ptr(new Gridmap(
            Config::Get<int>("grid_width"),
            Config::Get<int>("grid_height"),
            Config::Get<double>("grid_resolution"),
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
        association->SetGridmap(gridmap);
        if(use_imu)
            initializer->SetGridmap(gridmap);
        globalplanner=Global_planner::Ptr(new Global_planner(
            Config::Get<int>("grid_width"),
            Config::Get<int>("grid_height"),
            Config::Get<double>("grid_resolution")));
        frontend->SetGlobalPlanner(globalplanner);
        gridmap->SetGlobalPlanner(globalplanner);
        localplanner=Local_planner::Ptr(new Local_planner(
            Config::Get<int>("grid_width"),
            Config::Get<int>("grid_height"),
            Config::Get<double>("grid_resolution")));
        frontend->SetLocalPlanner(localplanner);
        gridmap->SetLocalPlanner(localplanner);
        globalplanner->SetLocalPlanner(localplanner);
    }
    return true;
}

std::map<FrontendStatus, std::string> map_status = {
    {FrontendStatus::BUILDING, "Building"},
    {FrontendStatus::INITIALIZING, "Initializing"},
    {FrontendStatus::TRACKING, "Tracking"},
    {FrontendStatus::LOST, "Lost"}};
void Estimator::InputImage(double time, cv::Mat &left_image, cv::Mat &right_image, SE3d init_odom)
{
    Frame::Ptr new_frame = Frame::Create();
    new_frame->time = time;
    new_frame->pose = init_odom;
    cv::undistort(left_image, new_frame->image_left, Camera::Get(0)->K, Camera::Get(0)->D);
    cv::undistort(right_image, new_frame->image_right, Camera::Get(1)->K, Camera::Get(1)->D);

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "Frontend status:" << map_status[frontend->status] << ", cost time: " << time_used.count() << " seconds.";
}

void Estimator::InputPointCloud(double time, Point3Cloud::Ptr point_cloud)
{
    auto t1 = std::chrono::steady_clock::now();
    association->AddScan(time, point_cloud);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    if (time_used.count() > 1e-2)
        LOG(INFO) << "Lidar Preprocessing cost time: " << time_used.count() << " seconds.";
}

void Estimator::InputImu(double time, Vector3d acc, Vector3d gyr)
{
    frontend->AddImu(time, acc, gyr);
}

void Estimator::InputNavSat(double time, double x, double y, double z, Vector3d cov)
{
    Navsat::Get()->AddPoint(time, x, y, z, cov);
}

} // namespace lvio_fusion
