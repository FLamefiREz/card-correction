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

struct bboxs {
    std::vector<std::vector<float>> detections;
    std::vector<int> ids;
};

struct Topks {
    std::vector<float> scores;
    std::vector<int> indexs;
};

struct _Topks {
    std::vector<float> scores;
    std::vector<int> indexs;
    std::vector<int> clses;
    std::vector<float> ys;
    std::vector<float> xs;
};

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b);
cv::Mat getAffineTransform(const cv::Point2f& center, float scale);
cv::Point2f transformPreds(cv::Point2f pt, cv::Point2f center, float scale);
std::vector<cv::Point2f> bbox_post_process(const std::vector<float>& box,
                                           cv::Point2f center, float scale);
std::vector<std::vector<float>> gather_feat(const ncnn::Mat& heat_mat, const std::vector<int>& inds);
Topks topk(const std::vector<float>& cls_scores, int k);
_Topks _topk(const ncnn::Mat& scores, int k);
ncnn::Mat equal(ncnn::Mat heatmap, const ncnn::Mat& pooled);
bboxs bbox_decode(const _Topks& peaks, const ncnn::Mat& corners, const ncnn::Mat& offsets,
                  const std::vector<int>& angles, const std::vector<int>& types, float scale);
std::vector<int> max_2dim(const std::vector<std::vector<float>>& classes);
std::vector<int> decode_by_ind(const ncnn::Mat& heat_mat, const std::vector<int>& inds);
float distance(float x1, float y1, float x2, float y2);
cv::Mat crop_image(const cv::Mat& image, const std::vector<cv::Point2f>& points);
cv::Mat crop_image(const cv::Mat& image, const std::vector<float>& box);
