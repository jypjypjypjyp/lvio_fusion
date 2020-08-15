#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include <pcl/kdtree/kdtree_flann.h>

namespace lvio_fusion
{
namespace lidar
{

const double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

void Feature::Associate(Lidar::Ptr lidar, std::shared_ptr<Frame> current_frame, std::shared_ptr<Frame> last_frame, ceres::Problem &problem)
{
    PointICloud &cornerPointsSharp = current_frame->feature_lidar->cornerPointsSharp;
    PointICloud &cornerPointsLessSharp = current_frame->feature_lidar->cornerPointsLessSharp;
    PointICloud &surfPointsFlat = current_frame->feature_lidar->surfPointsFlat;
    PointICloud &surfPointsLessFlat = current_frame->feature_lidar->surfPointsLessFlat;

    PointICloud &cornerPointsLessSharpLast = last_frame->feature_lidar->cornerPointsLessSharp;
    PointICloud &surfPointsLessFlatLast = last_frame->feature_lidar->surfPointsLessFlat;
    int cornerPointsSharpNum = cornerPointsSharp.points.size();
    int surfPointsFlatNum = surfPointsFlat.points.size();

    int corner_correspondence = 0;
    int plane_correspondence = 0;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

    pcl::KdTreeFLANN<PointI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointI>());
    pcl::KdTreeFLANN<PointI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointI>());
    kdtreeCornerLast->setInputCloud(PointICloud::Ptr(&cornerPointsLessSharpLast)); // 更新kdtree的点云
    kdtreeSurfLast->setInputCloud(PointICloud::Ptr(&surfPointsLessFlatLast));

    double *para_kf = current_frame->pose.data();

    pcl::PointXYZI pointSel;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // 基于最近邻原理建立corner特征点之间关联，find correspondence for corner features
    for (int i = 0; i < cornerPointsSharpNum; ++i)
    {
        pointSel = lidar->Transform(cornerPointsSharp.points[i], current_frame, last_frame); // 将当前帧的corner_sharp特征点O_cur，从当前帧的Lidar坐标系下变换到上一帧的Lidar坐标系下（记为点O，注意与前面的点O_cur不同），以利于寻找corner特征点的correspondence
        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);     // kdtree中的点云是上一帧的corner_less_sharp，所以这是在上一帧
                                                                                             // 的corner_less_sharp中寻找当前帧corner_sharp特征点O的最近邻点（记为A）

        int closestPointInd = -1, minPointInd2 = -1;
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) // 如果最近邻的corner特征点之间距离平方小于阈值，则最近邻点A有效
        {
            closestPointInd = pointSearchInd[0];
            int closestPointScanID = int(cornerPointsLessSharpLast.points[closestPointInd].intensity);

            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
            // 寻找点O的另外一个最近邻的点（记为点B） in the direction of increasing scan line
            for (int j = closestPointInd + 1; j < (int)cornerPointsLessSharpLast.points.size(); ++j) // cornerPointsLessSharpLast 来自上一帧的corner_less_sharp特征点,由于提取特征时是
            {                                                                                    // 按照scan的顺序提取的，所以cornerPointsLessSharpLast中的点也是按照scanID递增的顺序存放的
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
            Eigen::Vector3d curr_point(cornerPointsSharp.points[i].x,
                                       cornerPointsSharp.points[i].y,
                                       cornerPointsSharp.points[i].z);
            Eigen::Vector3d last_point_a(cornerPointsLessSharpLast.points[closestPointInd].x,
                                         cornerPointsLessSharpLast.points[closestPointInd].y,
                                         cornerPointsLessSharpLast.points[closestPointInd].z);
            Eigen::Vector3d last_point_b(cornerPointsLessSharpLast.points[minPointInd2].x,
                                         cornerPointsLessSharpLast.points[minPointInd2].y,
                                         cornerPointsLessSharpLast.points[minPointInd2].z);

            double s; // 运动补偿系数，kitti数据集的点云已经被补偿过，所以s = 1.0
            // 用点O，A，B构造点到线的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到直线AB的距离
            // 具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
            corner_correspondence++;
        }
    }
    // 下面说的点符号与上述相同
    // 与上面的建立corner特征点之间的关联类似，寻找平面特征点O的最近邻点ABC，即基于最近邻原理建立surf特征点之间的关联，find correspondence for plane features
    for (int i = 0; i < surfPointsFlatNum; ++i)
    {
        pointSel = lidar->Transform(surfPointsFlat.points[i], current_frame, last_frame); 
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

                Eigen::Vector3d curr_point(surfPointsFlat.points[i].x,
                                           surfPointsFlat.points[i].y,
                                           surfPointsFlat.points[i].z);
                Eigen::Vector3d last_point_a(surfPointsLessFlatLast.points[closestPointInd].x,
                                             surfPointsLessFlatLast.points[closestPointInd].y,
                                             surfPointsLessFlatLast.points[closestPointInd].z);
                Eigen::Vector3d last_point_b(surfPointsLessFlatLast.points[minPointInd2].x,
                                             surfPointsLessFlatLast.points[minPointInd2].y,
                                             surfPointsLessFlatLast.points[minPointInd2].z);
                Eigen::Vector3d last_point_c(surfPointsLessFlatLast.points[minPointInd3].x,
                                             surfPointsLessFlatLast.points[minPointInd3].y,
                                             surfPointsLessFlatLast.points[minPointInd3].z);

                // 用点O，A，B，C构造点到面的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到平面ABC的距离
                // 同样的，具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                plane_correspondence++;
            }
        }
    }
}
} // namespace lidar
} // namespace lvio_fusion