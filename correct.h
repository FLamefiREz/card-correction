#pragma once

#include "net.h"
#include <opencv2/core.hpp>

#include <filesystem>
#include <vector>

class CardCorrector {
public:
    explicit CardCorrector(const std::filesystem::path& model_directory);

    // Returns the input image when no detection meets the confidence threshold.
    cv::Mat correct(const cv::Mat& image) const;

private:
    ncnn::Net model_;
};

struct DecodedBoxes {
    // Each detection stores four (x, y) corners, then score, angle, and document type.
    std::vector<std::vector<float>> detections;
    std::vector<int> peak_indices;
};

struct RankedScores {
    std::vector<float> scores;
    std::vector<int> indices;
};

struct HeatmapPeaks {
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> class_indices;
    std::vector<float> y_coordinates;
    std::vector<float> x_coordinates;
};

cv::Point2f perpendicular_control_point(const cv::Point2f& a, const cv::Point2f& b);
cv::Mat heatmap_to_image_transform(const cv::Point2f& center, float scale);
cv::Point2f map_point_to_image(cv::Point2f point, cv::Point2f center, float scale);
std::vector<cv::Point2f> map_box_to_image(const std::vector<float>& box,
                                        cv::Point2f center, float scale);
std::vector<std::vector<float>> gather_score_rows(const ncnn::Mat& class_scores,
                                                 const std::vector<int>& indices);
RankedScores select_top_scores(const std::vector<float>& scores, int k);
HeatmapPeaks select_heatmap_peaks(const ncnn::Mat& scores, int k);
// Updates the shared heatmap buffer using the max-pooled scores.
ncnn::Mat suppress_non_maxima(ncnn::Mat heatmap, const ncnn::Mat& pooled);
DecodedBoxes decode_boxes(const HeatmapPeaks& peaks, const ncnn::Mat& corners, const ncnn::Mat& offsets,
                          const std::vector<int>& angles, const std::vector<int>& types, float scale);
std::vector<int> argmax_rows(const std::vector<std::vector<float>>& classes);
std::vector<int> decode_class_indices(const ncnn::Mat& class_scores, const std::vector<int>& indices);
float distance(float x1, float y1, float x2, float y2);
cv::Mat crop_image(const cv::Mat& image, const std::vector<cv::Point2f>& points);
cv::Mat crop_image(const cv::Mat& image, const std::vector<float>& box);
