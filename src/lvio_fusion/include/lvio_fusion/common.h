#ifndef lvio_fusion_COMMON_H
#define lvio_fusion_COMMON_H

// std
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// define the commonly included file to avoid a long include list
#define EIGEN_USE_BLAS
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

typedef Sophus::SE3d SE3d;
typedef Sophus::SO3d SO3d;

// opencv
#include <opencv2/opencv.hpp>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/common/io.h>

typedef pcl::PointXYZ Point3;
typedef typename pcl::PointCloud<Point3> Point3Cloud;
typedef pcl::PointXYZI PointI;
typedef typename pcl::PointCloud<PointI> PointICloud;
typedef pcl::PointXYZRGB PointRGB;
typedef typename pcl::PointCloud<PointRGB> PointRGBCloud;

// glog
#include <glog/logging.h>

// not implemented exception
class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented"){};
};

// system config
extern const double epsilon;
extern const int num_threads;

#endif // lvio_fusion_COMMON_H