#include "relocation.h"
#include <DBoW3/QueryResults.h>

namespace lvio_fusion
{

Relocation::Relocation(std::string voc_path)
{
    thread_ = std::thread(std::bind(&Relocation::RelocationLoop, this));
    detector_ = cv::ORB::create();
    voc_ = DBoW3::Vocabulary(voc_path);
    db_ = DBoW3::Database(voc_, false, 0);
    head_ = 0;
}

void Relocation::RelocationLoop()
{
    while (true)
    {
        while (head_ < frames_.size())
        {
            AddKeyFrameIntoVoc(frames_[head_]);
            DetectLoop(frames_[head_]);
            head_++;
        }
        std::chrono::milliseconds dura(100);
        std::this_thread::sleep_for(dura);
    }
}

void Relocation::AddKeyFrameIntoVoc(Frame::Ptr frame)
{
    static int thershold = 20;
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(frame->image_left, keypoints, 20, true);
    detector_->compute(frame->image_left, keypoints, frame->descriptors);
    db_.add(frame->descriptors);
}

void Relocation::DetectLoop(Frame::Ptr frame)
{
    //first query; then add this frame into database!
    DBoW3::QueryResults ret;
    db_.query(frame->descriptors, ret, 4, frame->id - 20);
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;
    // a good match with its nerghbour
    if (ret.size() >= 1 && ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
            }
        }
    
    if (find_loop && frame->id > 20)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        frame->loop = frames_[min_index];
    }
}

} // namespace lvio_fusion