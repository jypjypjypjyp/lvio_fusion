#include "lvio_fusion/dataset.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/config.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;

namespace lvio_fusion
{

Dataset::Dataset(const std::string &dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init()
{
    // read camera intrinsics and extrinsics
    cv::Mat cv_cam0_T_cam1 = Config::Get<cv::Mat>("cam0_T_cam1");
    Mat44 cam0_T_cam1;
    cv::cv2eigen(cv_cam0_T_cam1, cam0_T_cam1);
    Vec3 t(0, 0, 0);
    // first camera
    Camera::Ptr camera1(new Camera(Config::Get<double>("camera1.fx"),
                                   Config::Get<double>("camera1.fy"),
                                   Config::Get<double>("camera1.cx"),
                                   Config::Get<double>("camera1.cy"),
                                   t.norm(), SE3(SO3(), t)));
    cameras_.push_back(camera1);
    LOG(INFO) << "Camera 1"
              << " extrinsics: " << t.transpose();
    // second camera
    t << cam0_T_cam1(0, 3), cam0_T_cam1(1, 3), cam0_T_cam1(2, 3);
    Mat33 R(cam0_T_cam1.block(0, 0, 3, 3));
    Camera::Ptr camera2(new Camera(Config::Get<double>("camera2.fx"),
                                   Config::Get<double>("camera2.fy"),
                                   Config::Get<double>("camera2.cx"),
                                   Config::Get<double>("camera2.cy"),
                                   t.norm(), SE3(SO3(), t)));
    cameras_.push_back(camera2);
    LOG(INFO) << "Camera 2"
              << " extrinsics: " << t.transpose();
    current_image_index_ = 0;
    return true;
}

Frame::Ptr Dataset::NextFrame()
{
    boost::format fmt("%s/image_0%d/data/%010d.png");
    cv::Mat image_left, image_right;
    // read images
    image_left =
        cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr)
    {
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr;
    }

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left;
    new_frame->right_img_ = image_right;
    current_image_index_++;
    return new_frame;
}

} // namespace lvio_fusion