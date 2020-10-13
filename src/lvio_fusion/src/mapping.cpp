#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/utility.h"
#include <lvio_fusion/ceres/base.hpp>
#include <pcl/filters/voxel_grid.h>

namespace lvio_fusion
{
Mapping::Mapping()
{
    thread_ = std::thread(std::bind(&Mapping::MappingLoop, this));
}

inline void Mapping::AddToWorld(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out)
{
    double lti[7], lei[7], ltiei[7], cet[7];
    ceres::SE3Inverse(frame->pose.data(), lti);
    ceres::SE3Inverse(lidar_->extrinsic.data(), lei);
    ceres::SE3Product(lti, lei, ltiei);
    ceres::SE3Product(camera_->extrinsic.data(), frame->pose.data(), cet);

    for (int i = 0; i < in.size(); i++)
    {
        if (in[i].x <= 0)
            continue;
        //NOTE: Sophus is too slow
        // auto p_w = lidar_->Sensor2World(Vector3d(in[i].x, in[i].y, in[i].z), frame->pose);
        // auto pixel = camera_->World2Pixel(p_w, frame->pose);
        double pin[3] = {in[i].x, in[i].y, in[i].z}, pw[3], pc[3];
        ceres::SE3TransformPoint(ltiei, pin, pw);
        ceres::SE3TransformPoint(cet, pw, pc);
        auto pixel = camera_->Sensor2Pixel(Vector3d(pc[0], pc[1], pc[2]));

        auto &image = frame->image_left;
        if (0 < pixel.x() && pixel.x() < image.cols && 0 < pixel.y() && pixel.y() < image.rows)
        {
            unsigned char gray = image.at<uchar>((int)pixel.y(), (int)pixel.x());
            PointRGB point_world;
            point_world.x = pw[0];
            point_world.y = pw[1];
            point_world.z = pw[2];
            point_world.r = gray;
            point_world.g = gray;
            point_world.b = gray;
            out.push_back(point_world);
        }
    }
}

void Mapping::MappingLoop()
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        Frames &all_kfs = map_->GetAllKeyFrames();
        double active_time = map_->active_time;
        for (auto iter = all_kfs.upper_bound(head_); iter->first < active_time; iter++)
        {
            Frame::Ptr frame = iter->second;
            PointRGBCloud &map = map_->simple_map;
            AddToWorld(frame->feature_lidar->points_sharp, frame, map);
            AddToWorld(frame->feature_lidar->points_less_sharp, frame, map);
            AddToWorld(frame->feature_lidar->points_flat, frame, map);
            AddToWorld(frame->feature_lidar->points_less_flat, frame, map);
            frame->feature_lidar.reset();
            head_ = iter->first;
        }
        if (map_->simple_map.size() > 0)
        {
            Optimize();
        }
    }
}

void Mapping::Optimize()
{
    // lidar constraints
    ceres::LossFunction *lidar_loss_function = new ceres::HuberLoss(0.1);
    Frame::Ptr last_frame;
    Frame::Ptr current_frame;
    for (auto kf_pair : active_kfs)
    {
        if (kf_pair.second->feature_lidar)
        {
            current_frame = kf_pair.second;
            if (last_frame)
            {
                scan_registration_->Associate(current_frame, last_frame, problem, lidar_loss_function);
            }
            last_frame = current_frame;
        }
    }

    static pcl::VoxelGrid<PointRGB> downSizeFilter;
    PointRGBCloud::Ptr mapDS(new PointRGBCloud());
    pcl::copyPointCloud(map_->simple_map, *mapDS);
    downSizeFilter.setInputCloud(mapDS);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(map_->simple_map);
}

} // namespace lvio_fusion
