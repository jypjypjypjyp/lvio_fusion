
#ifndef lvio_fusion_VISUAL_ODOMETRY_H
#define lvio_fusion_VISUAL_ODOMETRY_H

#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/loop/relocator.h"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/navsat/navsat.h"

namespace lvio_fusion
{

enum Flag
{
    None = 0,
    Mono = 1,
    Stereo = 1 << 1,
    RGBD = 1 << 2,
    IMU = 1 << 3,
    Laser = 1 << 4,
    GNSS = 1 << 5,
};

class Estimator
{
public:
    typedef std::shared_ptr<Estimator> Ptr;

    Estimator(std::string &config_path);

    void InputImage(double time, cv::Mat &left_image, cv::Mat &right_image, SE3d init_odom);

    void InputNavSat(double time, double latitude, double longitude, double altitude, double posAccuracy);

    void InputPointCloud(double time, Point3Cloud::Ptr point_cloud);

    void InputImu(double time, Vector3d acc, Vector3d gyr);

    bool Init(int use_imu, int use_lidar, int use_navsat, int use_loop, int use_adapt);

    Frontend::Ptr frontend;
    Backend::Ptr backend;
    Relocator::Ptr relocator;
    FeatureAssociation::Ptr association;
    Mapping::Ptr mapping;
    Initializer::Ptr initializer;

private:
    std::string config_file_path_;
};
} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ODOMETRY_H
