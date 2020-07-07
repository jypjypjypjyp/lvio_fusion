#ifndef lvio_fusion_NAVSAT_H
#define lvio_fusion_NAVSAT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{
struct NavsatPoint
{
    NavsatPoint(double time, double x, double y, double z)
        : time(time), position(Vector3d(x, y, z)) {}
    double time;
    Vector3d position;
};

struct NavsatFrame
{
    double time;
    Vector3d A, B;
};

class Map;

class NavsatMap
{
public:
    typedef std::map<double, NavsatPoint> NavsatPoints;
    typedef std::unordered_map<double, NavsatFrame> NavsatFrames;
    typedef std::shared_ptr<NavsatMap> Ptr;

    NavsatMap(std::shared_ptr<Map> map) : map_(map) {}

    void AddPoint(NavsatPoint point)
    {
        navsat_points_.insert(std::make_pair(point.time, point));
    }

    NavsatPoints &GetAllPoints()
    {
        return navsat_points_;
    }

    NavsatFrame GetFrame(double t1, double t2)
    {
        // t1 < t2
        NavsatFrame frame;
        auto pair = navsat_frames_.find(t2);
        if (pair == navsat_frames_.end())
        {
            auto start = navsat_points_.lower_bound(t1);
            auto end = navsat_points_.lower_bound(t2);
            int n = std::distance(start, end) + 1;
            MatrixXd points(n, 3);
            auto iter = start;
            for (int i = 0; i < n; i++, iter++)
            {
                points.row(i) = (*iter).second.position;
            }
            frame.time = t2;
            line_fitting(points, frame.A, frame.B);
            navsat_frames_.insert(std::make_pair(t2, frame));
        }
        else
        {
            frame = (*pair).second;
        }
        return frame;
    }

    void Transfrom(NavsatFrame &frame)
    {
        Quaterniond q(R[0], R[1], R[2], R[3]);
        Vector3d t(t[0], t[1], t[2]);
        frame.A = q * frame.A + t;
        frame.B = q * frame.B + t;
    }

    void Initialize();

    bool initialized = false;
    int num_frames_init = 50;
    double R[4] = {1, 0, 0, 0}; // w, x, y, z
    double t[3] = {0, 0, 0};

private:
    NavsatPoints navsat_points_;
    NavsatFrames navsat_frames_;
    std::weak_ptr<Map> map_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_H