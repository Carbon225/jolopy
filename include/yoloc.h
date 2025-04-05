#ifndef YOLOC_H
#define YOLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct
{
    void *impl;
} yoloc_t;

typedef struct
{
    int class_id;
    float confidence;
    int x, y, w, h;
} yoloc_detection_t;

int yoloc_init(yoloc_t *yoloc, const char *model_path, int width, int height, int num_classes);

void yoloc_free(yoloc_t *yoloc);


void yoloc_set_confidence_threshold(yoloc_t *yoloc, float threshold);

float yoloc_get_confidence_threshold(yoloc_t *yoloc);


void yoloc_set_score_threshold(yoloc_t *yoloc, float threshold);

float yoloc_get_score_threshold(yoloc_t *yoloc);


void yoloc_set_nms_threshold(yoloc_t *yoloc, float threshold);

float yoloc_get_nms_threshold(yoloc_t *yoloc);


void yoloc_set_letterbox(yoloc_t *yoloc, int letterbox);

int yoloc_get_letterbox(yoloc_t *yoloc);


int yoloc_detect(yoloc_t *yoloc, const uint8_t *img, int width, int height, int channels, yoloc_detection_t *detections, int max_detections);

#ifdef __cplusplus
}
#endif

#endif
