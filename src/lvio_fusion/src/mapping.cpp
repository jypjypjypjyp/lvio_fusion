#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/utility.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace lvio_fusion
{
Mapping::Mapping()
{
    thread_ = std::thread(std::bind(&Mapping::MappingLoop, this));
}

inline void Mapping::AddToWorld(const PointICloud &in, Frame::Ptr frame, Point3Cloud &out)
{
    for (int i = 0; i < in.size(); i++)
    {
        auto p = lidar_->Sensor2World(Vector3d(in[i].x, in[i].y, in[i].z), frame->pose);
        out.push_back(Point3(p.x(), p.y(), p.z()));
    }
}

void Mapping::MappingLoop()
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Keyframes &all_kfs = map_->GetAllKeyFrames();
        double active_time = backend_->ActiveTime();
        for (auto iter = all_kfs.upper_bound(head_); iter->first < active_time; iter++)
        {
            Frame::Ptr frame = iter->second;
            Point3Cloud &map = map_->simple_map;
            AddToWorld(frame->feature_lidar->cornerPointsSharp, frame, map);
            AddToWorld(frame->feature_lidar->cornerPointsLessSharp, frame, map);
            AddToWorld(frame->feature_lidar->surfPointsFlat, frame, map);
            AddToWorld(frame->feature_lidar->surfPointsLessFlat, frame, map);
            head_ = iter->first;
        }
        Optimize();
    }
}

/**
 * remove points by indices
 * @param cloud_in      input
 * @param cloud_out     sorted indices
 */
template <typename PointT>
void remove_points_by_indices(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, const std::vector<int> &indices)
{
    for (int i : indices)
    {
        cloud_in[i].x = std::nan("0");
    }
    pcl::removeNaNFromPointCloud(cloud_in, cloud_out, std::vector<int>());
}

void Mapping::Optimize()
{
    static std::vector<pcl::ModelCoefficients> coefficients;
    Point3Cloud::Ptr map(&map_->simple_map);
    if (map->size())
    {
        return;
    }

    Point3Cloud outliers;
    pcl::ExtractIndices<Point3> extract;
    pcl::PointIndices::Ptr all_inliers(new pcl::PointIndices());
    pcl::ModelOutlierRemoval<Point3> model_filter;
    model_filter.setThreshold(0.01);
    model_filter.setModelType(pcl::SACMODEL_PLANE);
    model_filter.setInputCloud(map);
    for (auto coeff : coefficients)
    {
        std::vector<int> indices;
        model_filter.setModelCoefficients(coeff);
        model_filter.filter(indices);
        all_inliers->indices.insert(all_inliers->indices.end(), indices.begin(), indices.end());
    }
    extract.setInputCloud(map);
    extract.setIndices(all_inliers);
    extract.setNegative(true);
    extract.filter(outliers);
    // remove_points_by_indices(map, map, all_indices);

    pcl::PointIndices::Ptr new_inliers(new pcl::PointIndices());
    pcl::SACSegmentation<Point3> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    for (int i = 0; i < 5; i++)
    {
        pcl::ModelCoefficients coeff;
        seg.setInputCloud(Point3Cloud::Ptr(&outliers));
        seg.segment(*(new_inliers), coeff);
        if (new_inliers->indices.size() == 0)
        {
            break;
        }
        coefficients.push_back(coeff);
        all_inliers->indices.insert(all_inliers->indices.end(), new_inliers->indices.begin(), new_inliers->indices.end());
        extract.setInputCloud(Point3Cloud::Ptr(&outliers));
        extract.setIndices(new_inliers);
        extract.setNegative(true);
        extract.filter(outliers);
    }

    extract.setInputCloud(map);
    extract.setIndices(all_inliers);
    extract.setNegative(false);
    extract.filter(map_->simple_map);
}

} // namespace lvio_fusion
