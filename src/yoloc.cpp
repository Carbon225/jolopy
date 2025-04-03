#include "yoloc.h"

#include <vector>
#include <string>

#include "yolocpp.hpp"

using std::vector;
using std::string;

int yoloc_init(yoloc_t *yoloc)
{
    vector<string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    yoloc->impl = new YOLOCPP("yolov8n-quant.onnx", 640, 640, std::move(classes));
    if (!yoloc->impl)
    {
        return -1;
    }
    return 0;
}

void yoloc_free(yoloc_t *yoloc)
{
    if (yoloc->impl != nullptr)
    {
        delete static_cast<YOLOCPP*>(yoloc->impl);
        yoloc->impl = nullptr;
    }
}
