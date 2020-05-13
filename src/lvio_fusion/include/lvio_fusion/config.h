
#ifndef lvio_fusion_CONFIG_H
#define lvio_fusion_CONFIG_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Config
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {} // private conclassor makes a singleton
public:
    ~Config(); // close the file when deconclassing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    template <typename T>
    static T Get(const std::string &key)
    {
        T result;
        Config::config_->file_[key] >> result;
        return result;
    }
};
} // namespace lvio_fusion

#endif // lvio_fusion_CONFIG_H
