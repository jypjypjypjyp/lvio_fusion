#include "lvio_fusion/lidar/projection.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/voxel_grid.h>

namespace lvio_fusion
{

void ImageProjection::Clear()
{
    rangeMat = cv::Mat(num_scans_, horizon_scan_, CV_32F, cv::Scalar::all(FLT_MAX));
    groundMat = cv::Mat(num_scans_, horizon_scan_, CV_8S, cv::Scalar::all(0));
    labelMat = cv::Mat(num_scans_, horizon_scan_, CV_32S, cv::Scalar::all(0));
    labelCount = 1;
    PointI nanPoint; // fill in fullCloud at each iteration
    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;
    std::fill(points_full.points.begin(), points_full.points.end(), nanPoint);
}

SegmentedInfo ImageProjection::Process(PointICloud &points, PointICloud &points_segmented, PointICloud &points_outlier)
{
    SegmentedInfo segmented_info(num_scans_, horizon_scan_);

    FindStartEndAngle(segmented_info, points);

    ProjectPointCloud(segmented_info, points);

    RemoveGround(segmented_info);

    Segment(segmented_info, points_segmented, points_outlier);

    Clear();
    return segmented_info;
}

void ImageProjection::FindStartEndAngle(SegmentedInfo &segmented_info, PointICloud& points)
{
    // start and end orientation of this cloud
    segmented_info.start_orientation = -atan2(points.points[0].y, points.points[0].x);
    segmented_info.end_orientation = -atan2(points.points[points.points.size() - 1].y,
                                           points.points[points.points.size() - 1].x) +
                                    2 * M_PI;
    if (segmented_info.end_orientation - segmented_info.start_orientation > 3 * M_PI)
    {
        segmented_info.end_orientation -= 2 * M_PI;
    }
    else if (segmented_info.end_orientation - segmented_info.start_orientation < M_PI)
        segmented_info.end_orientation += 2 * M_PI;
    segmented_info.orientation_diff = segmented_info.end_orientation - segmented_info.start_orientation;
}

void ImageProjection::ProjectPointCloud(SegmentedInfo &segmented_info, PointICloud& points)
{
    // range image projection
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize;
    PointI thisPoint;

    cloudSize = points.points.size();

     for (size_t i = 0; i < cloudSize; ++i)
    {

        thisPoint.x = points.points[i].x;
        thisPoint.y = points.points[i].y;
        thisPoint.z = points.points[i].z;
        // find the row and column index in the image for this point

        verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
        rowIdn = (verticalAngle + ang_bottom_) / ang_res_y_;

        if (rowIdn < 0 || rowIdn >= num_scans_)
            continue;

        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        columnIdn = -round((horizonAngle - 90.0) / ang_res_x_) + horizon_scan_ / 2;
        if (columnIdn >= horizon_scan_)
            columnIdn -= horizon_scan_;

        if (columnIdn < 0 || columnIdn >= horizon_scan_)
            continue;

        range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);

        rangeMat.at<float>(rowIdn, columnIdn) = range;

        thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

        index = columnIdn + rowIdn * horizon_scan_;
        points_full.points[index] = thisPoint;
    }
}

void ImageProjection::RemoveGround(SegmentedInfo &segmented_info)
{
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    // groundMat
    // -1, no valid info to check if ground of not
    //  0, initial value, after validation, means not ground
    //  1, ground
    for (size_t j = 0; j < horizon_scan_; ++j)
    {
        for (size_t i = 0; i < ground_rows_; ++i)
        {

            lowerInd = j + (i)*horizon_scan_;
            upperInd = j + (i + 1) * horizon_scan_;

            if (points_full.points[lowerInd].intensity == -1 ||
                points_full.points[upperInd].intensity == -1)
            {
                // no info to check, invalid points
                groundMat.at<int8_t>(i, j) = -1;
                continue;
            }

            diffX = points_full.points[upperInd].x - points_full.points[lowerInd].x;
            diffY = points_full.points[upperInd].y - points_full.points[lowerInd].y;
            diffZ = points_full.points[upperInd].z - points_full.points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

            //NOTE: mount angle
            if (abs(angle) <= 10)
            {
                groundMat.at<int8_t>(i, j) = 1;
                groundMat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }
    // extract ground cloud (groundMat == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~num_scans_-1, need rangeMat for mark label matrix for the 16th scan
    for (size_t i = 0; i < num_scans_; ++i)
    {
        for (size_t j = 0; j < horizon_scan_; ++j)
        {
            if (groundMat.at<int8_t>(i, j) == 1 || rangeMat.at<float>(i, j) == FLT_MAX)
            {
                labelMat.at<int>(i, j) = -1;
            }
        }
    }
}

void ImageProjection::Segment(SegmentedInfo &segmented_info, PointICloud &points_segmented, PointICloud &points_outlier)
{
    // segmentation process
    for (size_t i = 0; i < num_scans_; ++i)
        for (size_t j = 0; j < horizon_scan_; ++j)
            if (labelMat.at<int>(i, j) == 0)
                LabelComponents(i, j);

    int sizeOfSegCloud = 0;
    // extract segmented cloud for lidar odometry
    for (size_t i = 0; i < num_scans_; ++i)
    {

        segmented_info.start_ring_index[i] = sizeOfSegCloud - 1 + 5;

        for (size_t j = 0; j < horizon_scan_; ++j)
        {
            if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1)
            {
                // outliers that will not be used for optimization (always continue)
                if (labelMat.at<int>(i, j) == 999999)
                {
                    if (i > ground_rows_ && j % 5 == 0)
                    {
                        points_outlier.push_back(points_full.points[j + i * horizon_scan_]);
                        continue;
                    }
                    else
                    {
                        continue;
                    }
                }
                // majority of ground points are skipped
                // if (groundMat.at<int8_t>(i, j) == 1)
                // {
                //     if (j % 5 != 0 && j > 5 && j < horizon_scan_ - 5)
                //         continue;
                // }
                // mark ground points so they will not be considered as edge features later
                segmented_info.ground_flag[sizeOfSegCloud] = (groundMat.at<int8_t>(i, j) == 1);
                // mark the points' column index for marking occlusion later
                segmented_info.col_ind[sizeOfSegCloud] = j;
                // save range info
                segmented_info.range[sizeOfSegCloud] = rangeMat.at<float>(i, j);
                // save seg cloud
                points_segmented.push_back(points_full.points[j + i * horizon_scan_]);
                // size of seg cloud
                ++sizeOfSegCloud;
            }
        }

        segmented_info.end_ring_index[i] = sizeOfSegCloud - 1 - 5;
    }
}

void ImageProjection::LabelComponents(int row, int col)
{
    static uint16_t *allPushedIndX = new uint16_t[num_scans_ * horizon_scan_]; // array for tracking points of a segmented object
    static uint16_t *allPushedIndY = new uint16_t[num_scans_ * horizon_scan_];
    static uint16_t *queueIndX = new uint16_t[num_scans_ * horizon_scan_]; // array for breadth-first search process of segmentation, for speed
    static uint16_t *queueIndY = new uint16_t[num_scans_ * horizon_scan_];
    static std::vector<std::pair<int8_t, int8_t>> neighborIterator; // neighbor iterator for segmentaiton process
    if (neighborIterator.empty())
    {
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = 1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = -1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);
    }

    // use std::queue std::vector std::deque will slow the program down greatly
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY;
    std::vector<bool> lineCountFlag(num_scans_, false);

    queueIndX[0] = row;
    queueIndY[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;

    while (queueSize > 0)
    {
        // Pop point
        fromIndX = queueIndX[queueStartInd];
        fromIndY = queueIndY[queueStartInd];
        --queueSize;
        ++queueStartInd;
        // Mark popped point
        labelMat.at<int>(fromIndX, fromIndY) = labelCount;
        // Loop through all the neighboring grids of popped grid
        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
        {
            // new index
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;
            // index should be within the boundary
            if (thisIndX < 0 || thisIndX >= num_scans_)
                continue;
            // at range image margin (left or right side)
            if (thisIndY < 0)
                thisIndY = horizon_scan_ - 1;
            if (thisIndY >= horizon_scan_)
                thisIndY = 0;
            // prevent infinite loop (caused by put already examined point back)
            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                continue;

            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                          rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                          rangeMat.at<float>(thisIndX, thisIndY));

            if ((*iter).first == 0)
                alpha = segment_alpha_x_;
            else
                alpha = segment_alpha_y_;

            angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

            if (angle > theta)
            {

                queueIndX[queueEndInd] = thisIndX;
                queueIndY[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;

                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;

                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;
            }
        }
    }

    // check if this segment is valid
    bool feasibleSegment = false;
    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= num_segment_valid_points_)
    {
        int lineCount = 0;
        for (size_t i = 0; i < num_scans_; ++i)
            if (lineCountFlag[i] == true)
                ++lineCount;
        if (lineCount >= num_segment_valid_lines_)
            feasibleSegment = true;
    }
    // segment is valid, mark these points
    if (feasibleSegment == true)
    {
        ++labelCount;
    }
    else
    { // segment is invalid, mark these points
        for (size_t i = 0; i < allPushedIndSize; ++i)
        {
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
}

} // namespace lvio_fusion