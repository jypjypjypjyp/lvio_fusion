#ifndef lvio_fusion_LOCAL_H
#define lvio_fusion_LOCAL_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{
namespace visual
{
class LocalMapping
{
public:
    void AddKeyFrame(Frame::Ptr new_kf);
private:
    void LocalBA(Frame::Ptr new_kf);
    void RemoveOldFeame();
    void Search();
};
} // namespace visual
} // namespace lvio_fusion

#endif //!lvio_fusion_LOCAL_H
