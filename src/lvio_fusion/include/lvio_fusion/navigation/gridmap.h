#ifndef lvio_fusion_GRIDMAP_H
#define lvio_fusion_GRIDMAP_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/navigation/global_planner.h"

namespace lvio_fusion
{
class LaserScan
{
public:
    typedef std::shared_ptr<LaserScan> Ptr;
    LaserScan(Frame::Ptr& frame_,std::vector<Vector2d> scan_points_)
        :frame(frame_),scan_points(scan_points_)
    {
    }
    Frame::Ptr GetFrame(){return frame;}
    std::vector<Vector2d> GetPoints(){return scan_points;}
private:
    Frame::Ptr frame;
    std::vector<Vector2d> scan_points;
};

class Gridmap
{
public:
    typedef std::shared_ptr<Gridmap> Ptr;

    Gridmap(int width_, int height_, double resolution_, int num_scans, int horizon_scan, double ang_res_y, double ang_bottom, int ground_rows, double cycle_time, double min_range, double max_range, double deskew, double spacing)
        :width(width_), height(height_), resolution(resolution_),  num_scans_(num_scans), horizon_scan_(horizon_scan),
          ang_res_x_(360.0 / float(horizon_scan)), ang_res_y_(ang_res_y), ang_bottom_(ang_bottom),
          ground_rows_(ground_rows),
          segment_alpha_x_(ang_res_x_ / 180.0 * M_PI), segment_alpha_y_(ang_res_y_ / 180.0 * M_PI),
          cycle_time_(cycle_time), min_range_(min_range), max_range_(max_range), deskew_(deskew), spacing_(spacing)
    {
        InitGridmap();
        ClearMap();
    }

    void InitGridmap()
    {
        grid_map.create(height, width, CV_32FC1);
	    grid_map_int.create(height, width, CV_8SC1);
        visual_counter.create(height, width, CV_32SC1);
        occupied_counter.create(height, width, CV_32SC1);
    }

    void ClearMap(){
        grid_map_int.setTo(-1);
        grid_map.setTo(0);
        visual_counter.setTo(0);
        occupied_counter.setTo(0);
    }

    void AddScan(PointICloud scan_msg);

    void AddFrame(Frame::Ptr& frame);

    void ToCartesianCoordinates(PointICloud scan_msg,Frame::Ptr& frame);
    
    void Bresenhamline (double x1,double y1,double x2,double y2, std::vector<Eigen::Vector2i>& points);
    
    void SetGlobalPlanner(Global_planner::Ptr globalplanner ) { globalplanner_ = globalplanner; }//NAVI
    
    void SetLocalPlanner(Local_planner::Ptr localplanner ) { localplanner_ = localplanner; }//NAVI

    cv::Mat GetGridmap();

    Vector2i  GetIndex(int x, int y);

    int width;
    int height;
    double resolution;
private:
    cv::Mat grid_map_int;
    cv::Mat grid_map;
    cv::Mat visual_counter;
    cv::Mat occupied_counter;

    Frame::Ptr current_frame;
    std::vector<LaserScan::Ptr> laser_scans_2d;
    Global_planner::Ptr globalplanner_;//NAVI
    Local_planner::Ptr localplanner_;//NAVI

    const int num_scans_;
    const int horizon_scan_;
    const float ang_res_x_;
    const float ang_res_y_;
    const float ang_bottom_;
    const int ground_rows_;

    const float theta = 60.0 / 180.0 * M_PI; // decrese this value may improve accuracy
    const int num_segment_valid_points_ = 5;
    const int num_segment_valid_lines_ = 3;
    const float segment_alpha_x_;
    const float segment_alpha_y_;

    const double cycle_time_;
    const double min_range_;
    const double max_range_;
    const bool deskew_;
    const double spacing_;
    
};

} // namespace lvio_fusion
#endif // lvio_fusion_GRIDMAP_H