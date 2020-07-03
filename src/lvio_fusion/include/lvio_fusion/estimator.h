
#ifndef lvio_fusion_VISUAL_ODOMETRY_H
#define lvio_fusion_VISUAL_ODOMETRY_H

#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frontend.h"
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

    //TODO
    void InputPointCloud(double time, PointCloudI::Ptr point_cloud);

    //TODO
    void InputIMU(double time, Vector3d acc, Vector3d gyr);

    bool Init();

    Frontend::Ptr frontend = nullptr;
    Backend::Ptr backend = nullptr;
    Map::Ptr map = nullptr;

private:
    bool inited_ = false;
    std::string config_file_path_;
};
} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ODOMETRY_H
