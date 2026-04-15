#ifndef STEREO_APP_ZED_CAPTURE_BRIDGE_H
#define STEREO_APP_ZED_CAPTURE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zed_capture_handle zed_capture_handle;

typedef struct zed_frame_buffer {
    uint64_t frame_idx;
    uint64_t timestamp_ns;
    int width;
    int height;
    int channels;
    uint8_t* left_data;
    size_t left_size;
    uint8_t* right_data;
    size_t right_size;
} zed_frame_buffer;

zed_capture_handle* zed_capture_create(void);
void zed_capture_destroy(zed_capture_handle* handle);

int zed_capture_open(
    zed_capture_handle* handle,
    const char* resolution,
    int fps,
    const char* color_space,
    char* error_message,
    size_t error_message_size
);

void zed_capture_close(zed_capture_handle* handle);

int zed_capture_get_calibration(
    zed_capture_handle* handle,
    float out_k[9],
    float* out_baseline_m,
    int* out_width,
    int* out_height,
    char* error_message,
    size_t error_message_size
);

int zed_capture_grab(
    zed_capture_handle* handle,
    zed_frame_buffer* out_frame,
    char* error_message,
    size_t error_message_size
);

void zed_capture_release_frame(zed_frame_buffer* frame);

#ifdef __cplusplus
}
#endif

#endif
