#include "../correct.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc == 2) {
        const std::filesystem::path root(argv[1]);
        const CardCorrector corrector(root);
        const auto input = cv::imread((root / "example/1.jpg").string());
        const auto corrected = corrector.correct(input);
        if (corrected.cols < 3600 || corrected.cols > 4400 ||
            corrected.rows < 2250 || corrected.rows > 2750) {
            std::cerr << "Sample must be cropped and output in landscape orientation\n";
            return 1;
        }

        const cv::Mat blank = cv::Mat::zeros(160, 240, CV_8UC3);
        const auto unchanged = corrector.correct(blank);
        if (unchanged.size() != blank.size() || cv::norm(unchanged, blank, cv::NORM_INF) != 0) {
            std::cerr << "No-detection fallback must keep the original image\n";
            return 1;
        }
        return 0;
    }

    int failures = 0;
    const auto expect = [&failures](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            ++failures;
        }
    };

    ncnn::Mat classes(2, 2);
    classes.row(0)[0] = 0.9f;
    classes.row(0)[1] = 0.1f;
    classes.row(1)[0] = 0.2f;
    classes.row(1)[1] = 0.8f;
    expect(decode_by_ind(classes, {0, 1}) == std::vector<int>({0, 1}),
           "Each detection must use its own classification scores");

    ncnn::Mat heatmap(3, 2);
    heatmap.fill(0.0f);
    heatmap.row(0)[2] = 0.9f;
    heatmap.row(1)[2] = 0.8f;
    ncnn::Mat pooled(3, 2);
    pooled.fill(0.9f);
    const auto masked = equal(heatmap, pooled);
    expect(masked.row(1)[2] == 0.0f, "Peak masking must suppress nonmaxima");
    const auto peak = _topk(masked, 1);
    expect(peak.indexs == std::vector<int>{2} && peak.xs == std::vector<float>{2} &&
               peak.ys == std::vector<float>{0},
           "Peak coordinates must use the actual row width");
    expect(_topk(masked, 20).indexs.size() == 6 && _topk(masked, 0).indexs.empty(),
           "Top-K must respect the number of available scores");

    const cv::Point2f third = get3rdPoint({100, 50}, {100, 0});
    const float cross = 50 * (third.x - 100);
    expect(std::abs(cross) > 1.0f, "Affine control points must not be collinear");
    if (std::abs(cross) > 1.0f) {
        const cv::Point2f center = transformPreds({96, 96}, {100, 50}, 100);
        expect(cv::norm(center - cv::Point2f(100, 50)) < 0.01,
               "The output center must map back to the original center");
        const cv::Point2f point = transformPreds({48, 48}, {100, 50}, 100);
        expect(cv::norm(point - cv::Point2f(75, 25)) < 0.01,
               "The inverse affine transform must preserve scale and axes");
    }

    const cv::Mat image(24, 32, CV_8UC3, cv::Scalar(12, 34, 56));
    bool rejected = false;
    try {
        crop_image(image, std::vector<cv::Point2f>(4, {4, 4}));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "Degenerate crop must be rejected before warping");

    rejected = false;
    try {
        crop_image(image, std::vector<float>{100, 100, 108, 100, 108, 106, 100, 106});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "Crop entirely outside the image must be rejected");

    const auto partial_crop = crop_image(image, std::vector<float>{-3, 4, 5, 4, 5, 10, -3, 10});
    expect(partial_crop.size() == cv::Size(8, 6), "Crop may extend partially beyond the image");

    const cv::Mat crop = crop_image(image, std::vector<float>{3, 4, 11, 4, 11, 10, 3, 10});
    expect(crop.size() == cv::Size(8, 6), "Valid quadrilateral must retain its dimensions");
    expect(crop.at<cv::Vec3b>(2, 2) == cv::Vec3b(12, 34, 56),
           "Valid crop must sample the original image");

    return failures == 0 ? 0 : 1;
}
