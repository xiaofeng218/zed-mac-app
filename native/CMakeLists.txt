cmake_minimum_required(VERSION 3.20)

project(zed_capture_bridge LANGUAGES CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-fobjc-arc)

find_package(CURL REQUIRED)

set(ZED_OPEN_CAPTURE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/zed-open-capture-mac)
set(ZED_OPEN_CAPTURE_INCLUDE ${ZED_OPEN_CAPTURE_ROOT}/include)
set(ZED_OPEN_CAPTURE_SRC ${ZED_OPEN_CAPTURE_ROOT}/src)

add_library(zed_capture_bridge SHARED
    ${ZED_OPEN_CAPTURE_SRC}/zed_calibration_data.cpp
    ${ZED_OPEN_CAPTURE_SRC}/zed_video_capture.mm
    ${ZED_OPEN_CAPTURE_SRC}/ZEDVideoCapture.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/zed_capture_bridge.cpp
)

target_include_directories(zed_capture_bridge
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZED_OPEN_CAPTURE_INCLUDE}
    ${ZED_OPEN_CAPTURE_SRC}
)

target_link_libraries(zed_capture_bridge
    PRIVATE
    "-framework Foundation"
    "-framework AVFoundation"
    "-framework CoreMedia"
    "-framework CoreVideo"
    "-framework CoreGraphics"
    "-framework Accelerate"
    "-framework IOKit"
    CURL::libcurl
)
