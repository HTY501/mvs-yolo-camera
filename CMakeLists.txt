cmake_minimum_required(VERSION 3.10)
project(mvs_yolo)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
# 自动查找系统安装的 OpenVINO
find_package(OpenVINO REQUIRED)

# MVS SDK
set(MVS_ROOT "/opt/MVS")
include_directories(${MVS_ROOT}/include)
link_directories(${MVS_ROOT}/lib/64)

add_executable(mvs_yolo main.cpp)

target_link_libraries(mvs_yolo
    ${OpenCV_LIBS}
    MvCameraControl
    openvino::runtime # 链接 OpenVINO 运行时
    pthread
    X11
)
