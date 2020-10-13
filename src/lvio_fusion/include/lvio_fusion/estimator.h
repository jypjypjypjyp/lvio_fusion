
#ifndef lvio_fusion_VISUAL_ODOMETRY_H
#define lvio_fusion_VISUAL_ODOMETRY_H

#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/lidar/scan_registration.h"
#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/loop/relocation.h"
#include "lvio_fusion/semantic/detected_object.h"

namespace lvio_fusion
{

class Estimator
{
public:
    typedef std::shared_ptr<Estimator> Ptr;

    Estimator(std::string &config_path);

    void InputImage(double time, cv::Mat &left_image, cv::Mat &right_image, std::vector<DetectedObject> objects = {});

    void InputNavSat(double time, double latitude, double longitude, double altitude, double posAccuracy);

    void InputPointCloud(double time, Point3Cloud::Ptr point_cloud);

    void InputIMU(double time, Vector3d acc, Vector3d gyr);

    bool Init(int use_imu, int use_lidar, int use_navsat, int use_loop, int is_semantic);

    Map::Ptr map;
    Frontend::Ptr frontend;
    Backend::Ptr backend;
    Relocation::Ptr relocation;
    ScanRegistration::Ptr scan_registration;
    Mapping::Ptr mapping;
    Initializer::Ptr initializer;

private:
    std::string config_file_path_;
};
} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ODOMETRY_H
