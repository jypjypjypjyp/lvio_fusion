#ifndef lvio_fusion_RELOCATION_H
#define lvio_fusion_RELOCATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include <DBoW3/DBoW3.h>
#include <DBoW3/Database.h>
#include <DBoW3/Vocabulary.h>

namespace lvio_fusion
{

class Relocation
{
public:
    typedef std::shared_ptr<Relocation> Ptr;

    Relocation(std::string voc_path);

    void AddFrame(Frame::Ptr frame)
    {
        frames_.push_back(frame);
    }

private:
    void RelocationLoop();

    bool DetectLoop(Frame::Ptr frame);

    void AddKeyFrameIntoVoc(Frame::Ptr frame);

    void Associate(Frame::Ptr frame);
    
    std::thread thread_;
    cv::Ptr<cv::Feature2D> detector_;
    std::vector<Frame::Ptr> frames_;
    int head_;
    DBoW3::Database db_;
    DBoW3::Vocabulary voc_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H