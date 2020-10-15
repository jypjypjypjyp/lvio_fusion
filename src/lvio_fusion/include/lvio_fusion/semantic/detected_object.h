#ifndef lvio_fusion_DETECTED_OBJECT_H
#define lvio_fusion_DETECTED_OBJECT_H

namespace lvio_fusion
{

enum class LabelType
{
    None,
    Car,
    Person,
    Truck
};

class DetectedObject
{
public:
    DetectedObject(){};
    DetectedObject(LabelType label, float p, int xmin, int ymin, int xmax, int ymax)
    :label(label), probability(p), xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax)
    {};

    LabelType label;
    float probability;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};
} // namespace lvio_fusion

#endif // lvio_fusion_DETECTED_OBJECT_H