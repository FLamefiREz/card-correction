// Created by 钟顺民 on 2024/4/30.

#include "correct.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
void print_usage(std::ostream& stream) {
    stream << "Usage: cvDetectionCorrection [--no-gui] [input-image [output-image [model-dir]]]\n"
              "Defaults: example/1.jpg, result.jpg, and the current directory or model/.\n";
}

std::filesystem::path default_model_directory() {
    if (std::filesystem::is_regular_file("model/correction.param") &&
        std::filesystem::is_regular_file("model/correction.bin")) {
        return "model";
    }
    return ".";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_usage(std::cout);
        return 0;
    }
    const bool show_windows = argc == 1 || std::string_view(argv[1]) != "--no-gui";
    const int first_path = show_windows ? 1 : 2;
    const int path_count = argc - first_path;
    if (path_count > 3) {
        print_usage(std::cerr);
        return 2;
    }

    const char* input_path = path_count >= 1 ? argv[first_path] : "example/1.jpg";
    const char* output_path = path_count >= 2 ? argv[first_path + 1] : "result.jpg";

    try {
        const auto image = cv::imread(input_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error(std::string("Could not read image: ") + input_path);
        }

        const std::filesystem::path model_directory =
            path_count >= 3 ? std::filesystem::path(argv[first_path + 2]) : default_model_directory();
        const CardCorrector corrector(model_directory);
        const auto result = corrector.correct(image);
        if (!cv::imwrite(output_path, result)) {
            throw std::runtime_error(std::string("Could not write image: ") + output_path);
        }

        std::cout << "Saved " << output_path << '\n';
        if (show_windows) {
            cv::imshow("src", image);
            cv::imshow("result", result);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
