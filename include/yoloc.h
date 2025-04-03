#ifndef YOLOC_H
#define YOLOC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    void *impl;
} yoloc_t;

int yoloc_init(yoloc_t *yoloc);

void yoloc_free(yoloc_t *yoloc);

#ifdef __cplusplus
}
#endif

#endif
