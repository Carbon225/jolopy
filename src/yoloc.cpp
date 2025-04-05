#include "yoloc.h"

#include <vector>
#include <string>

#include "yolocpp.hpp"

using std::vector;
using std::string;

int yoloc_init(yoloc_t *yoloc, const char *model_path, int width, int height, int num_classes)
{
    vector<string> classes;
    for (int i = 0; i < num_classes; i++)
    {
        classes.push_back(std::to_string(i));
    }
    try
    {
        yoloc->impl = new YOLOCPP(model_path, width, height, std::move(classes));
    }
    catch (const std::exception &e)
    {
        yoloc_free(yoloc);
        return -1;
    }
    if (yoloc->impl == nullptr) return -1;
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

void yoloc_set_confidence_threshold(yoloc_t *yoloc, float threshold)
{
    if (yoloc->impl == nullptr) return;
    static_cast<YOLOCPP*>(yoloc->impl)->confidence_threshold = threshold;
}

float yoloc_get_confidence_threshold(yoloc_t *yoloc)
{
    if (yoloc->impl == nullptr) return 0.0f;
    return static_cast<YOLOCPP*>(yoloc->impl)->confidence_threshold;
}

void yoloc_set_score_threshold(yoloc_t *yoloc, float threshold)
{
    if (yoloc->impl == nullptr) return;
    static_cast<YOLOCPP*>(yoloc->impl)->score_threshold = threshold;
}

float yoloc_get_score_threshold(yoloc_t *yoloc)
{
    if (yoloc->impl == nullptr) return 0.0f;
    return static_cast<YOLOCPP*>(yoloc->impl)->score_threshold;
}

void yoloc_set_nms_threshold(yoloc_t *yoloc, float threshold)
{
    if (yoloc->impl == nullptr) return;
    static_cast<YOLOCPP*>(yoloc->impl)->nms_threshold = threshold;
}

float yoloc_get_nms_threshold(yoloc_t *yoloc)
{
    if (yoloc->impl == nullptr) return 0.0f;
    return static_cast<YOLOCPP*>(yoloc->impl)->nms_threshold;
}

void yoloc_set_letterbox(yoloc_t *yoloc, int letterbox)
{
    if (yoloc->impl == nullptr) return;
    static_cast<YOLOCPP*>(yoloc->impl)->letterbox = letterbox;
}

int yoloc_get_letterbox(yoloc_t *yoloc)
{
    if (yoloc->impl == nullptr) return 0;
    return static_cast<YOLOCPP*>(yoloc->impl)->letterbox;
}

int yoloc_detect(yoloc_t *yoloc, const uint8_t *img, int width, int height, int channels, yoloc_detection_t *detections, int max_detections)
{
    if (yoloc->impl == nullptr) return 0;
    try
    {
        const auto all_detections = static_cast<YOLOCPP*>(yoloc->impl)->detect(img, width, height, channels);
        int num_detections = 0;
        for (const auto &detection : all_detections)
        {
            if (num_detections >= max_detections)
            {
                break;
            }
            detections[num_detections].class_id = detection.class_id;
            detections[num_detections].confidence = detection.confidence;
            detections[num_detections].x = detection.x;
            detections[num_detections].y = detection.y;
            detections[num_detections].w = detection.w;
            detections[num_detections].h = detection.h;
            num_detections++;
        }
        return num_detections;
    }
    catch (const std::exception &e)
    {
        return -1;
    }
}
