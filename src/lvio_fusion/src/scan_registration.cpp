#include "lvio_fusion/lidar/scan_registration.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/utility.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace lvio_fusion
{
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

void ScanRegistration::AddScan(double time, Point3Cloud::Ptr new_scan)
{
    raw_point_clouds_.insert(std::make_pair(time, new_scan));

    Keyframes &all_kfs = map_->GetAllKeyFrames();
    for (auto iter = all_kfs.upper_bound(head_); iter != all_kfs.end() && iter->first < time; iter++)
    {
        PointICloud point_cloud;
        if (TimeAlign(iter->first, point_cloud))
        {
            Preprocess(point_cloud, iter->second);
            head_ = iter->first;
        }
    }
}

bool ScanRegistration::TimeAlign(double time, PointICloud &out)
{
    auto iter = raw_point_clouds_.upper_bound(time);
    Point3Cloud &pc2 = *(iter->second);
    double end_time = iter->first + cycle_time_ / 2;
    Point3Cloud &pc1 = *((--iter)->second);
    double start_time = iter->first - cycle_time_ / 2;
    Point3Cloud pc = pc1 + pc2;
    int size = pc.size();
    if (time - cycle_time_ / 2 < start_time || time + cycle_time_ / 2 > end_time)
    {
        return false;
    }
    auto start_iter = pc.begin() + size * (time - start_time - cycle_time_ / 2) / (end_time - start_time);
    auto end_iter = pc.begin() + size * (time - start_time + cycle_time_ / 2) / (end_time - start_time);
    Point3Cloud out3;
    out3.clear();
    out3.insert(out3.begin(), start_iter, end_iter);
    pcl::copyPointCloud(out3, out);
    return true;
}

void ScanRegistration::UndistortPoint(PointI &p, Frame::Ptr frame)
{
    double time_delta = (p.intensity - int(p.intensity));
    double time = frame->time - cycle_time_ * 0.5 + time_delta;
    SE3d pose = map_->ComputePose(time);
    auto p1 = lidar_->Sensor2World(Vector3d(p.x, p.y, p.z), pose);
    auto p2 = lidar_->World2Sensor(p1, frame->pose);
    p.x = p2.x();
    p.y = p2.y();
    p.z = p2.z();
}

inline void ScanRegistration::UndistortPointCloud(PointICloud &pc, Frame::Ptr frame)
{
    for (auto &p : pc)
    {
        UndistortPoint(p, frame);
    }
}

inline void ScanRegistration::Deskew(Frame::Ptr frame)
{
    frame->feature_lidar->cornerPointsSharpDeskew = frame->feature_lidar->cornerPointsSharp;
    frame->feature_lidar->cornerPointsLessSharpDeskew = frame->feature_lidar->cornerPointsLessSharp;
    frame->feature_lidar->surfPointsFlatDeskew = frame->feature_lidar->surfPointsFlat;
    frame->feature_lidar->surfPointsLessFlatDeskew = frame->feature_lidar->surfPointsLessFlat;
    UndistortPointCloud(frame->feature_lidar->cornerPointsSharpDeskew, frame);
    UndistortPointCloud(frame->feature_lidar->cornerPointsLessSharpDeskew, frame);
    UndistortPointCloud(frame->feature_lidar->surfPointsFlatDeskew, frame);
    UndistortPointCloud(frame->feature_lidar->surfPointsLessFlatDeskew, frame);
}

void ScanRegistration::Preprocess(PointICloud &pc, Frame::Ptr frame)
{
    PointICloud cornerPointsSharp;
    PointICloud cornerPointsLessSharp;
    PointICloud surfPointsFlat;
    PointICloud surfPointsLessFlat;

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(pc, pc, indices);
    remove_close_points(pc, pc, minimum_range_);

    int raw_size = pc.size();
    float start_ori = -atan2(pc.points[0].y, pc.points[0].x);
    float end_ori = -atan2(pc.points[raw_size - 1].y, pc.points[raw_size - 1].x) + 2 * M_PI;

    if (end_ori - start_ori > 3 * M_PI)
    {
        end_ori -= 2 * M_PI;
    }
    else if (end_ori - start_ori < M_PI)
    {
        end_ori += 2 * M_PI;
    }

    int size = raw_size;
    std::vector<PointICloud> scans_in_point_cloud(num_scans_);
    PointI point;
    bool half_passed = false;
    for (int i = 0; i < raw_size; i++)
    {
        point = pc[i];
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scan_id = 0;
        if (num_scans_ == 16)
        {
            scan_id = int((angle + 15) / 2 + 0.5);
            if (scan_id > (num_scans_ - 1) || scan_id < 0)
            {
                size--;
                continue;
            }
        }
        else if (num_scans_ == 32)
        {
            scan_id = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
            if (scan_id > (num_scans_ - 1) || scan_id < 0)
            {
                size--;
                continue;
            }
        }
        else if (num_scans_ == 64)
        {
            if (angle >= -8.83)
                scan_id = int((2 - angle) * 3.0 + 0.5);
            else
                scan_id = num_scans_ / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies
            if (angle > 2 || angle < -24.33 || scan_id > 50 || scan_id < 0)
            {
                size--;
                continue;
            }
        }

        float ori = -atan2(point.y, point.x);
        if (!half_passed)
        {
            if (ori < start_ori - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > start_ori + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - start_ori > M_PI)
            {
                half_passed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < end_ori - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > end_ori + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        point.intensity = scan_id + cycle_time_ * (ori - start_ori) / (end_ori - start_ori);
        scans_in_point_cloud[scan_id].push_back(point);
    }

    pc.clear();

    std::vector<int> start_index(num_scans_, 0);
    std::vector<int> end_index(num_scans_, 0);
    PointICloud::Ptr laserCloud(new PointICloud());
    for (int i = 0; i < num_scans_; i++)
    {
        start_index[i] = laserCloud->size() + 5; // 记录每个scan的开始index，忽略前5个点
        *laserCloud += scans_in_point_cloud[i];
        end_index[i] = laserCloud->size() - 6; // 记录每个scan的结束index，忽略后5个点，开始和结束处的点云scan容易产生不闭合的“接缝”，对提取edge feature不利
    }

    static float cloudCurvature[150000];
    static int cloudSortInd[150000];
    static int cloudNeighborPicked[150000];
    static int cloudLabel[150000];
    for (int i = 5; i < size - 5; i++)
    {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0; // 点有没有被选选择为feature点
        cloudLabel[i] = 0;          // Label 2: corner_sharp
                                    // Label 1: corner_less_sharp, 包含Label 2
                                    // Label -1: surf_flat
                                    // Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样
    }

    for (int i = 0; i < num_scans_; i++) // 按照scan的顺序提取4种特征点
    {
        if (end_index[i] - start_index[i] < 6) // 如果该scan的点数少于7个点，就跳过
            continue;
        PointICloud::Ptr surfPointsLessFlatScan(new PointICloud());
        for (int j = 0; j < 6; j++) // 将该scan分成6小段执行特征检测
        {
            int sp = start_index[i] + (end_index[i] - start_index[i]) * j / 6;           // subscan的起始index
            int ep = start_index[i] + (end_index[i] - start_index[i]) * (j + 1) / 6 - 1; // subscan的结束index

            std::sort(cloudSortInd + sp, cloudSortInd + ep + 1,
                      [](int i, int j) {
                          return (cloudCurvature[i] < cloudCurvature[j]);
                      }); // 根据曲率有小到大对subscan的点进行sort

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--) // 从后往前，即从曲率大的点开始提取corner feature
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1) // 如果该点没有被选择过，并且曲率大于0.1
                {
                    largestPickedNum++;
                    if (largestPickedNum <= 2) // 该subscan中曲率最大的前2个点认为是corner_sharp特征点
                    {
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 20) // 该subscan中曲率最大的前20个点认为是corner_less_sharp特征点
                    {
                        cloudLabel[ind] = 1;
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; // 标记该点被选择过了

                    // 与当前点距离的平方 <= 0.05的点标记为选择过，避免特征点密集分布
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 提取surf平面feature，与上述类似，选取该subscan曲率最小的前4个点为surf_flat
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1;
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 其他的非corner特征点与surf_flat特征点一起组成surf_less_flat特征点
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        // 最后对该scan点云中提取的所有surf_less_flat特征点进行降采样，因为点太多了
        PointICloud surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointI> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat = surfPointsLessFlatScanDS;
    }
    lidar::Feature::Ptr feature = lidar::Feature::Create(cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat);
    frame->feature_lidar = feature;
}

inline void ScanRegistration::Transform(const PointI &in, Frame::Ptr from, Frame::Ptr to, PointI &out)
{
    auto p1 = lidar_->Sensor2World(Vector3d(in.x, in.y, in.z), from->pose);
    auto p2 = lidar_->World2Sensor(p1, to->pose);
    PointI r;
    out.x = p2.x();
    out.y = p2.y();
    out.z = p2.z();
    out.intensity = in.intensity;
}

void ScanRegistration::Associate(Frame::Ptr current_frame, Frame::Ptr last_frame, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    if (deskew_)
    {
        assert(last_frame->feature_lidar->iterations >= current_frame->feature_lidar->iterations);
        if (last_frame->feature_lidar->iterations == current_frame->feature_lidar->iterations)
        {
            Deskew(last_frame);
            last_frame->feature_lidar->iterations++;
        }
        Deskew(current_frame);
        current_frame->feature_lidar->iterations = last_frame->feature_lidar->iterations;
    }
    PointICloud &cornerPointsSharp = current_frame->feature_lidar->cornerPointsSharpDeskew;
    PointICloud &cornerPointsLessSharp = current_frame->feature_lidar->cornerPointsLessSharpDeskew;
    PointICloud &surfPointsFlat = current_frame->feature_lidar->surfPointsFlatDeskew;
    PointICloud &surfPointsLessFlat = current_frame->feature_lidar->surfPointsLessFlatDeskew;
    PointICloud &cornerPointsLessSharpLast = last_frame->feature_lidar->cornerPointsLessSharpDeskew;
    PointICloud &surfPointsLessFlatLast = last_frame->feature_lidar->surfPointsLessFlatDeskew;

    int cornerPointsSharpNum = cornerPointsSharp.points.size();
    int surfPointsFlatNum = surfPointsFlat.points.size();

    pcl::KdTreeFLANN<PointI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointI>());
    pcl::KdTreeFLANN<PointI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointI>());
    kdtreeCornerLast->setInputCloud(PointICloud::Ptr(&cornerPointsLessSharpLast)); // 更新kdtree的点云
    kdtreeSurfLast->setInputCloud(PointICloud::Ptr(&surfPointsLessFlatLast));

    double *para_kf = current_frame->pose.data();
    double *para_last_kf = last_frame->pose.data();

    PointI pointSel;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // 基于最近邻原理建立corner特征点之间关联，find correspondence for corner features
    for (int i = 0; i < cornerPointsSharpNum; ++i)
    {
        Transform(cornerPointsSharp.points[i], current_frame, last_frame, pointSel);     // 将当前帧的corner_sharp特征点O_cur，从当前帧的Lidar坐标系下变换到上一帧的Lidar坐标系下（记为点O，注意与前面的点O_cur不同），以利于寻找corner特征点的correspondence
        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); // kdtree中的点云是上一帧的corner_less_sharp，所以这是在上一帧
                                                                                         // 的corner_less_sharp中寻找当前帧corner_sharp特征点O的最近邻点（记为A）

        int closestPointInd = -1, minPointInd2 = -1;
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) // 如果最近邻的corner特征点之间距离平方小于阈值，则最近邻点A有效
        {
            closestPointInd = pointSearchInd[0];
            int closestPointScanID = int(cornerPointsLessSharpLast.points[closestPointInd].intensity);

            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
            // 寻找点O的另外一个最近邻的点（记为点B） in the direction of increasing scan line
            for (int j = closestPointInd + 1; j < (int)cornerPointsLessSharpLast.points.size(); ++j) // cornerPointsLessSharpLast 来自上一帧的corner_less_sharp特征点,由于提取特征时是
            {                                                                                        // 按照scan的顺序提取的，所以cornerPointsLessSharpLast中的点也是按照scanID递增的顺序存放的
                // if in the same scan line, continue
                if (int(cornerPointsLessSharpLast.points[j].intensity) <= closestPointScanID) // intensity整数部分存放的是scanID
                    continue;

                // if not in nearby scans, end the loop
                if (int(cornerPointsLessSharpLast.points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                    break;

                double pointSqDis = (cornerPointsLessSharpLast.points[j].x - pointSel.x) *
                                        (cornerPointsLessSharpLast.points[j].x - pointSel.x) +
                                    (cornerPointsLessSharpLast.points[j].y - pointSel.y) *
                                        (cornerPointsLessSharpLast.points[j].y - pointSel.y) +
                                    (cornerPointsLessSharpLast.points[j].z - pointSel.z) *
                                        (cornerPointsLessSharpLast.points[j].z - pointSel.z);

                if (pointSqDis < minPointSqDis2) // 第二个最近邻点有效,，更新点B
                {
                    // find nearer point
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            }

            // 寻找点O的另外一个最近邻的点B in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j)
            {
                // if in the same scan line, continue
                if (int(cornerPointsLessSharpLast.points[j].intensity) >= closestPointScanID)
                    continue;

                // if not in nearby scans, end the loop
                if (int(cornerPointsLessSharpLast.points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                    break;

                double pointSqDis = (cornerPointsLessSharpLast.points[j].x - pointSel.x) *
                                        (cornerPointsLessSharpLast.points[j].x - pointSel.x) +
                                    (cornerPointsLessSharpLast.points[j].y - pointSel.y) *
                                        (cornerPointsLessSharpLast.points[j].y - pointSel.y) +
                                    (cornerPointsLessSharpLast.points[j].z - pointSel.z) *
                                        (cornerPointsLessSharpLast.points[j].z - pointSel.z);

                if (pointSqDis < minPointSqDis2) // 第二个最近邻点有效，更新点B
                {
                    // find nearer point
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            }
        }
        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
        {                      // 即特征点O的两个最近邻点A和B都有效
            Vector3d curr_point(cornerPointsSharp.points[i].x,
                                cornerPointsSharp.points[i].y,
                                cornerPointsSharp.points[i].z);
            Vector3d last_point_a(cornerPointsLessSharpLast.points[closestPointInd].x,
                                  cornerPointsLessSharpLast.points[closestPointInd].y,
                                  cornerPointsLessSharpLast.points[closestPointInd].z);
            Vector3d last_point_b(cornerPointsLessSharpLast.points[minPointInd2].x,
                                  cornerPointsLessSharpLast.points[minPointInd2].y,
                                  cornerPointsLessSharpLast.points[minPointInd2].z);

            // 用点O，A，B构造点到线的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到直线AB的距离
            // 具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
            ceres::CostFunction *cost_function = LidarEdgeError::Create(curr_point, last_point_a, last_point_b, lidar_);
            problem.AddResidualBlock(cost_function, loss_function, para_last_kf, para_kf);
        }
    }
    // 下面说的点符号与上述相同
    // 与上面的建立corner特征点之间的关联类似，寻找平面特征点O的最近邻点ABC，即基于最近邻原理建立surf特征点之间的关联，find correspondence for plane features
    for (int i = 0; i < surfPointsFlatNum; ++i)
    {
        Transform(surfPointsFlat.points[i], current_frame, last_frame, pointSel);
        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) // 找到的最近邻点A有效
        {
            closestPointInd = pointSearchInd[0];

            // get closest point's scan ID
            int closestPointScanID = int(surfPointsLessFlatLast.points[closestPointInd].intensity);
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

            // search in the direction of increasing scan line
            for (int j = closestPointInd + 1; j < (int)surfPointsLessFlatLast.points.size(); ++j)
            {
                // if not in nearby scans, end the loop
                if (int(surfPointsLessFlatLast.points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                    break;

                double pointSqDis = (surfPointsLessFlatLast.points[j].x - pointSel.x) *
                                        (surfPointsLessFlatLast.points[j].x - pointSel.x) +
                                    (surfPointsLessFlatLast.points[j].y - pointSel.y) *
                                        (surfPointsLessFlatLast.points[j].y - pointSel.y) +
                                    (surfPointsLessFlatLast.points[j].z - pointSel.z) *
                                        (surfPointsLessFlatLast.points[j].z - pointSel.z);

                // if in the same or lower scan line
                if (int(surfPointsLessFlatLast.points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                {
                    minPointSqDis2 = pointSqDis; // 找到的第2个最近邻点有效，更新点B，注意如果scanID准确的话，一般点A和点B的scanID相同
                    minPointInd2 = j;
                }
                // if in the higher scan line
                else if (int(surfPointsLessFlatLast.points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                {
                    minPointSqDis3 = pointSqDis; // 找到的第3个最近邻点有效，更新点C，注意如果scanID准确的话，一般点A和点B的scanID相同,且与点C的scanID不同，与LOAM的paper叙述一致
                    minPointInd3 = j;
                }
            }

            // search in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j)
            {
                // if not in nearby scans, end the loop
                if (int(surfPointsLessFlatLast.points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                    break;

                double pointSqDis = (surfPointsLessFlatLast.points[j].x - pointSel.x) *
                                        (surfPointsLessFlatLast.points[j].x - pointSel.x) +
                                    (surfPointsLessFlatLast.points[j].y - pointSel.y) *
                                        (surfPointsLessFlatLast.points[j].y - pointSel.y) +
                                    (surfPointsLessFlatLast.points[j].z - pointSel.z) *
                                        (surfPointsLessFlatLast.points[j].z - pointSel.z);

                // if in the same or higher scan line
                if (int(surfPointsLessFlatLast.points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                {
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
                else if (int(surfPointsLessFlatLast.points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                {
                    // find nearer point
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }

            if (minPointInd2 >= 0 && minPointInd3 >= 0) // 如果三个最近邻点都有效
            {

                Vector3d curr_point(surfPointsFlat.points[i].x,
                                    surfPointsFlat.points[i].y,
                                    surfPointsFlat.points[i].z);
                Vector3d last_point_a(surfPointsLessFlatLast.points[closestPointInd].x,
                                      surfPointsLessFlatLast.points[closestPointInd].y,
                                      surfPointsLessFlatLast.points[closestPointInd].z);
                Vector3d last_point_b(surfPointsLessFlatLast.points[minPointInd2].x,
                                      surfPointsLessFlatLast.points[minPointInd2].y,
                                      surfPointsLessFlatLast.points[minPointInd2].z);
                Vector3d last_point_c(surfPointsLessFlatLast.points[minPointInd3].x,
                                      surfPointsLessFlatLast.points[minPointInd3].y,
                                      surfPointsLessFlatLast.points[minPointInd3].z);

                // 用点O，A，B，C构造点到面的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到平面ABC的距离
                // 同样的，具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
                ceres::CostFunction *cost_function = LidarPlaneError::Create(curr_point, last_point_a, last_point_b, last_point_c, lidar_);
                problem.AddResidualBlock(cost_function, loss_function, para_last_kf, para_kf);
            }
        }
    }
}
} // namespace lvio_fusion
