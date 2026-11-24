cmake_minimum_required(VERSION 3.27)
project(cvDetectionCorrection LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(ncnn CONFIG REQUIRED)

add_library(card_correction STATIC correct.cpp)
target_compile_features(card_correction PUBLIC cxx_std_17)
target_link_libraries(card_correction PUBLIC ncnn opencv_core opencv_imgproc)

add_executable(cvDetectionCorrection main.cpp)
target_link_libraries(cvDetectionCorrection PRIVATE card_correction opencv_imgcodecs opencv_highgui)

include(CTest)
if(BUILD_TESTING)
    add_executable(correction_tests tests/correction_tests.cpp)
    target_link_libraries(correction_tests PRIVATE card_correction opencv_imgcodecs)
    add_test(NAME correction_unit COMMAND correction_tests)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/correction.param"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/correction.bin"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example/1.jpg")
        add_test(NAME correction_sample COMMAND correction_tests "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()
