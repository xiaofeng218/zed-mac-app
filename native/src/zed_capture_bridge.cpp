#include "zed_capture_bridge.h"

#include "zed_calibration_data.h"
#include "zed_video_capture.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace zed;

struct CalibrationSnapshot {
    std::array<float, 9> k {};
    float baseline_m = 0.0f;
    int width = 0;
    int height = 0;
};

struct LatestFrame {
    uint64_t frame_idx = 0;
    uint64_t timestamp_ns = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> left;
    std::vector<uint8_t> right;
};

struct CaptureState {
    VideoCapture capture;
    StereoDimensions dimensions;
    CalibrationSnapshot calibration;
    LatestFrame latest_frame;
    std::mutex mutex;
    bool is_open = false;
    bool is_started = false;
    bool has_frame = false;
    std::string color_space = "RGB";
};

void write_error(char* buffer, size_t buffer_size, const std::string& message) {
    if (buffer == nullptr || buffer_size == 0) {
        return;
    }
    const size_t copy_size = std::min(buffer_size - 1, message.size());
    std::memcpy(buffer, message.data(), copy_size);
    buffer[copy_size] = '\0';
}

std::string calibration_section_for_side(const std::string& resolution_string, bool left_side) {
    return std::string(left_side ? "LEFT_CAM_" : "RIGHT_CAM_") + resolution_string;
}

CalibrationSnapshot read_calibration(VideoCapture& capture, StereoDimensions dimensions) {
    CalibrationData calibration_data = capture.getCalibrationData();
    const std::string resolution_string = calibration_data.calibrationString(dimensions);
    const std::string left_section = calibration_section_for_side(resolution_string, true);

    CalibrationSnapshot snapshot;
    snapshot.k = {
        calibration_data.get<float>(left_section, "fx"), 0.0f, calibration_data.get<float>(left_section, "cx"),
        0.0f, calibration_data.get<float>(left_section, "fy"), calibration_data.get<float>(left_section, "cy"),
        0.0f, 0.0f, 1.0f,
    };
    snapshot.baseline_m = calibration_data.get<float>("STEREO", "Baseline") / 1000.0f;
    snapshot.width = dimensions.width / 2;
    snapshot.height = dimensions.height;
    return snapshot;
}

void split_stereo_frame(const uint8_t* data, size_t height, size_t width, size_t channels, LatestFrame& frame) {
    const size_t single_width = width / 2;
    const size_t row_bytes = width * channels;
    const size_t single_row_bytes = single_width * channels;

    frame.width = static_cast<int>(single_width);
    frame.height = static_cast<int>(height);
    frame.channels = static_cast<int>(channels);
    frame.left.resize(single_row_bytes * height);
    frame.right.resize(single_row_bytes * height);

    for (size_t row = 0; row < height; ++row) {
        const uint8_t* row_start = data + (row * row_bytes);
        std::memcpy(frame.left.data() + (row * single_row_bytes), row_start, single_row_bytes);
        std::memcpy(frame.right.data() + (row * single_row_bytes), row_start + single_row_bytes, single_row_bytes);
    }
}

StereoDimensions open_with_runtime_settings(
    VideoCapture& capture,
    const std::string& resolution,
    int fps,
    const std::string& color_space
) {
    const auto parse_color = [&]() -> ColorSpace {
        if (color_space == "RGB") {
            return RGB;
        }
        if (color_space == "BGR") {
            return BGR;
        }
        if (color_space == "GREYSCALE") {
            return GREYSCALE;
        }
        if (color_space == "YUV") {
            return YUV;
        }
        throw std::runtime_error("Unsupported color space: " + color_space);
    };

    const ColorSpace parsed_color = parse_color();

    if (resolution == "HD2K" && fps == 15) {
        return capture.open<HD2K, FPS_15>(parsed_color);
    }
    if (resolution == "HD1080" && fps == 15) {
        return capture.open<HD1080, FPS_15>(parsed_color);
    }
    if (resolution == "HD1080" && fps == 30) {
        return capture.open<HD1080, FPS_30>(parsed_color);
    }
    if (resolution == "HD720" && fps == 15) {
        return capture.open<HD720, FPS_15>(parsed_color);
    }
    if (resolution == "HD720" && fps == 30) {
        return capture.open<HD720, FPS_30>(parsed_color);
    }
    if (resolution == "HD720" && fps == 60) {
        return capture.open<HD720, FPS_60>(parsed_color);
    }
    if (resolution == "VGA" && fps == 15) {
        return capture.open<VGA, FPS_15>(parsed_color);
    }
    if (resolution == "VGA" && fps == 30) {
        return capture.open<VGA, FPS_30>(parsed_color);
    }
    if (resolution == "VGA" && fps == 60) {
        return capture.open<VGA, FPS_60>(parsed_color);
    }
    if (resolution == "VGA" && fps == 100) {
        return capture.open<VGA, FPS_100>(parsed_color);
    }

    throw std::runtime_error("Unsupported resolution/FPS combination: " + resolution + " @ " + std::to_string(fps));
}

}  // namespace

struct zed_capture_handle {
    CaptureState state;
};

extern "C" {

zed_capture_handle* zed_capture_create(void) {
    return new zed_capture_handle();
}

void zed_capture_destroy(zed_capture_handle* handle) {
    if (handle == nullptr) {
        return;
    }
    zed_capture_close(handle);
    delete handle;
}

int zed_capture_open(
    zed_capture_handle* handle,
    const char* resolution,
    int fps,
    const char* color_space,
    char* error_message,
    size_t error_message_size
) {
    if (handle == nullptr) {
        write_error(error_message, error_message_size, "Capture handle is null");
        return -1;
    }

    try {
        auto& state = handle->state;
        if (state.is_open) {
            throw std::runtime_error("Capture is already open");
        }

        const std::string resolution_string = resolution == nullptr ? "HD720" : resolution;
        state.color_space = color_space == nullptr ? "RGB" : color_space;
        state.dimensions = open_with_runtime_settings(state.capture, resolution_string, fps, state.color_space);
        state.calibration = read_calibration(state.capture, state.dimensions);
        state.capture.start([&state](uint8_t* data, size_t height, size_t width, size_t channels) {
            LatestFrame frame;
            split_stereo_frame(data, height, width, channels, frame);
            frame.frame_idx = state.latest_frame.frame_idx + 1;
            frame.timestamp_ns = static_cast<uint64_t>(
                std::chrono::time_point_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now()
                ).time_since_epoch().count()
            );

            std::lock_guard<std::mutex> lock(state.mutex);
            if (state.color_space == "BGR" && frame.channels == 3) {
                for (auto& pixel : {std::ref(frame.left), std::ref(frame.right)}) {
                    auto& bytes = pixel.get();
                    for (size_t i = 0; i + 2 < bytes.size(); i += 3) {
                        std::swap(bytes[i], bytes[i + 2]);
                    }
                }
            }
            state.latest_frame = std::move(frame);
            state.has_frame = true;
        });
        state.is_open = true;
        state.is_started = true;
        return 0;
    }
    catch (const std::exception& error) {
        write_error(error_message, error_message_size, error.what());
        return -1;
    }
}

void zed_capture_close(zed_capture_handle* handle) {
    if (handle == nullptr) {
        return;
    }

    auto& state = handle->state;
    if (!state.is_open) {
        return;
    }

    if (state.is_started) {
        state.capture.stop();
        state.is_started = false;
    }
    state.capture.close();
    state.is_open = false;
    state.has_frame = false;
}

int zed_capture_get_calibration(
    zed_capture_handle* handle,
    float out_k[9],
    float* out_baseline_m,
    int* out_width,
    int* out_height,
    char* error_message,
    size_t error_message_size
) {
    if (handle == nullptr) {
        write_error(error_message, error_message_size, "Capture handle is null");
        return -1;
    }

    try {
        const auto& calibration = handle->state.calibration;
        if (!handle->state.is_open) {
            throw std::runtime_error("Capture is not open");
        }
        std::copy(calibration.k.begin(), calibration.k.end(), out_k);
        *out_baseline_m = calibration.baseline_m;
        *out_width = calibration.width;
        *out_height = calibration.height;
        return 0;
    }
    catch (const std::exception& error) {
        write_error(error_message, error_message_size, error.what());
        return -1;
    }
}

int zed_capture_grab(
    zed_capture_handle* handle,
    zed_frame_buffer* out_frame,
    char* error_message,
    size_t error_message_size
) {
    if (handle == nullptr || out_frame == nullptr) {
        write_error(error_message, error_message_size, "Capture handle or output frame is null");
        return -1;
    }

    try {
        auto& state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.is_open) {
            throw std::runtime_error("Capture is not open");
        }
        if (!state.has_frame) {
            return 0;
        }

        const LatestFrame& frame = state.latest_frame;
        out_frame->frame_idx = frame.frame_idx;
        out_frame->timestamp_ns = frame.timestamp_ns;
        out_frame->width = frame.width;
        out_frame->height = frame.height;
        out_frame->channels = frame.channels;
        out_frame->left_size = frame.left.size();
        out_frame->right_size = frame.right.size();
        out_frame->left_data = new uint8_t[out_frame->left_size];
        out_frame->right_data = new uint8_t[out_frame->right_size];
        std::memcpy(out_frame->left_data, frame.left.data(), out_frame->left_size);
        std::memcpy(out_frame->right_data, frame.right.data(), out_frame->right_size);
        return 1;
    }
    catch (const std::exception& error) {
        write_error(error_message, error_message_size, error.what());
        return -1;
    }
}

void zed_capture_release_frame(zed_frame_buffer* frame) {
    if (frame == nullptr) {
        return;
    }
    delete[] frame->left_data;
    delete[] frame->right_data;
    frame->left_data = nullptr;
    frame->right_data = nullptr;
    frame->left_size = 0;
    frame->right_size = 0;
}

}  // extern "C"
