cmake_minimum_required(VERSION 3.27)
project(cvDetectionCorrection)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_TOOLCHAIN_FILE "D:\\project\\project\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")
set(OpenCV_DIR "D:/project/project/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)
#include_directories( ${OpenCV_INCLUDE_DIRS})

set(ncnn_DIR ncnn-20240410-windows-vs2022/x64/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_executable(cvDetectionCorrection main.cpp
        correct.cpp)

target_link_libraries(cvDetectionCorrection
        ncnn
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_freetype
)