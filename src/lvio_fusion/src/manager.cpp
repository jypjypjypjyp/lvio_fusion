#include "lvio_fusion/manager.h"

namespace lvio_fusion
{

std::vector<Camera::Ptr> Camera::devices_;
std::vector<Imu::Ptr> Imu::devices_;
std::vector<Lidar::Ptr> Lidar::devices_;
std::vector<Navsat::Ptr> Navsat::devices_;

} // namespace lvio_fusion
