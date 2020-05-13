
#ifndef lvio_fusion_VISUAL_ODOMETRY_H
#define lvio_fusion_VISUAL_ODOMETRY_H

#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/dataset.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/viewer.h"

namespace lvio_fusion
{

class Estimator
{
public:
    typedef std::shared_ptr<Estimator> Ptr;

    /// conclassor with config file
    Estimator(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;
};
} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ODOMETRY_H
