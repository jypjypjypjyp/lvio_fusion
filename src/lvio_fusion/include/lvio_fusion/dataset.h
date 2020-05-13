#ifndef lvio_fusion_DATASET_H
#define lvio_fusion_DATASET_H
#include "lvio_fusion/camera.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

class Dataset
{
public:
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string &dataset_path);

    bool Init();

    /// create and return the next frame containing the stereo images
    Frame::Ptr NextFrame();

    /// get camera by id
    Camera::Ptr GetCamera(int camera_id) const
    {
        return cameras_.at(camera_id);
    }

private:
    std::string dataset_path_;
    int current_image_index_ = 0;

    std::vector<Camera::Ptr> cameras_;
};
} // namespace lvio_fusion

#endif