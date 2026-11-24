// Created by 钟顺民 on 2024/4/30.
// 单据、证件矫正

#include "correct.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using std::vector;

namespace {
constexpr int input_size = 768;
constexpr int model_stride = 4;
constexpr int heatmap_size = input_size / model_stride;
constexpr float confidence_threshold = 0.3f;

bool is_float_tensor(const ncnn::Mat& tensor) {
    return !tensor.empty() && tensor.elempack == 1 && tensor.elemsize == sizeof(float);
}

bool is_heatmap(const ncnn::Mat& tensor) {
    return is_float_tensor(tensor) && (tensor.dims == 2 || tensor.dims == 3) && tensor.c == 1;
}

void require_head(const ncnn::Mat& head, int columns, int rows, const char* name) {
    if (!is_float_tensor(head) || head.dims != 2 || head.w != columns || head.h != rows) {
        throw std::runtime_error(std::string("Unexpected model output shape or type: ") + name);
    }
}

ncnn::Mat extract(ncnn::Extractor& extractor, const char* name) {
    ncnn::Mat output;
    if (extractor.extract(name, output) != 0 || output.empty()) {
        throw std::runtime_error(std::string("Could not extract model output: ") + name);
    }
    return output;
}

vector<cv::Point2f> corners_from_box(const vector<float>& box) {
    if (box.size() < 8) {
        throw std::invalid_argument("A crop requires four corner coordinates");
    }
    return {{box[0], box[1]}, {box[2], box[3]}, {box[4], box[5]}, {box[6], box[7]}};
}
}

CardCorrector::CardCorrector(const std::filesystem::path& model_directory) {
    const auto param_path = (model_directory / "correction.param").string();
    const auto weights_path = (model_directory / "correction.bin").string();
    if (model_.load_param(param_path.c_str()) != 0) {
        throw std::runtime_error("Could not load model parameters: " + param_path);
    }
    if (model_.load_model(weights_path.c_str()) != 0) {
        throw std::runtime_error("Could not load model weights: " + weights_path);
    }
}

cv::Mat CardCorrector::correct(const cv::Mat& image) const {
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("Correction requires a nonempty 8-bit BGR image");
    }

    // Center the source in model coordinates without allocating a full-size square image.
    const double square_size = std::max(image.cols, image.rows);
    const double x_offset = (square_size - image.cols) / 2;
    const double y_offset = (square_size - image.rows) / 2;
    const double input_scale = input_size / square_size;
    const cv::Matx23d transform(input_scale, 0, x_offset * input_scale,
                                0, input_scale, y_offset * input_scale);
    cv::Mat resized;
    cv::warpAffine(image, resized, transform, cv::Size(input_size, input_size));

    const float means[] = {0.408f * 255, 0.447f * 255, 0.470f * 255};
    const float norms[] = {1 / (0.289f * 255), 1 / (0.274f * 255), 1 / (0.278f * 255)};
    auto input = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, resized.cols, resized.rows);
    if (input.empty()) {
        throw std::runtime_error("Could not allocate model input");
    }
    input.substract_mean_normalize(means, norms);

    auto extractor = model_.create_extractor();
    if (extractor.input("in0", input) != 0) {
        throw std::runtime_error("Could not set model input: in0");
    }

    auto heatmap = extract(extractor, "/Sigmoid_output_0");
    const auto pooled = extract(extractor, "out0");
    if (!is_heatmap(heatmap) || heatmap.w != heatmap_size || heatmap.h != heatmap_size) {
        throw std::runtime_error("Unexpected model heatmap shape or type");
    }
    const auto peaks = _topk(equal(heatmap, pooled), 1);
    if (peaks.scores.front() < confidence_threshold) {
        return image;
    }

    const int positions = heatmap.w * heatmap.h;
    const auto corners = extract(extractor, "out1");
    const auto angles = extract(extractor, "out2");
    const auto types = extract(extractor, "out3");
    const auto offsets = extract(extractor, "out4");
    require_head(corners, 8, positions, "out1");
    require_head(angles, 4, positions, "out2");
    require_head(types, 2, positions, "out3");
    require_head(offsets, 2, positions, "out4");

    const auto direction = decode_by_ind(angles, peaks.indexs);
    const auto document_type = decode_by_ind(types, peaks.indexs);
    const auto boxes = bbox_decode(peaks, corners, offsets, direction, document_type,
                                   static_cast<float>(1 / input_scale));
    auto points = corners_from_box(boxes.detections.front());
    for (auto& point : points) {
        point.x -= static_cast<float>(x_offset);
        point.y -= static_cast<float>(y_offset);
    }

    auto result = crop_image(image, points);
    switch (direction.front()) {
    case 1:
        cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
    case 2:
        cv::rotate(result, result, cv::ROTATE_180);
        break;
    case 3:
        cv::rotate(result, result, cv::ROTATE_90_CLOCKWISE);
        break;
    }
    return result;
}

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b) {
    const auto direction = a - b;
    return b + cv::Point2f(-direction.y, direction.x);
}

cv::Mat getAffineTransform(const cv::Point2f& center, float scale) {
    const float half_heatmap = heatmap_size / 2.0f;
    cv::Point2f source[] = {center, center + cv::Point2f(0, -scale / 2), {}};
    cv::Point2f destination[] = {{half_heatmap, half_heatmap}, {half_heatmap, 0}, {}};
    source[2] = get3rdPoint(source[0], source[1]);
    destination[2] = get3rdPoint(destination[0], destination[1]);
    return cv::getAffineTransform(destination, source);
}

cv::Point2f transformPreds(cv::Point2f pt, cv::Point2f center, float scale) {
    const cv::Matx23d transform = getAffineTransform(center, scale);
    const auto point = transform * cv::Vec3d(pt.x, pt.y, 1);
    return {static_cast<float>(point[0]), static_cast<float>(point[1])};
}

vector<cv::Point2f> bbox_post_process(const vector<float>& box, cv::Point2f center, float scale) {
    auto points = corners_from_box(box);
    cv::transform(points, points, getAffineTransform(center, scale));
    return points;
}

vector<vector<float>> gather_feat(const ncnn::Mat& heat_mat, const vector<int>& inds) {
    if (!is_float_tensor(heat_mat) || heat_mat.dims != 2) {
        throw std::invalid_argument("Expected a matrix of classification scores");
    }
    vector<vector<float>> rows;
    rows.reserve(inds.size());
    for (const int index : inds) {
        if (index < 0 || index >= heat_mat.h) {
            throw std::out_of_range("Classification row is outside the tensor");
        }
        const float* row = heat_mat.row(index);
        rows.emplace_back(row, row + heat_mat.w);
    }
    return rows;
}

Topks topk(const vector<float>& cls_scores, int k) {
    const int count = static_cast<int>(cls_scores.size());
    k = std::clamp(k, 0, count);
    if (k == 0) {
        return {};
    }

    vector<std::pair<float, int>> ranked;
    ranked.reserve(count);
    for (int i = 0; i < count; ++i) {
        ranked.emplace_back(cls_scores[i], i);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), std::greater<>{});

    Topks result;
    result.scores.reserve(k);
    result.indexs.reserve(k);
    for (int i = 0; i < k; ++i) {
        result.scores.push_back(ranked[i].first);
        result.indexs.push_back(ranked[i].second);
    }
    return result;
}

_Topks _topk(const ncnn::Mat& scores, int k) {
    if (!is_heatmap(scores) || k <= 0) {
        return {};
    }

    const float* values = scores.row(0);
    const int count = scores.w * scores.h;
    Topks ranked;
    if (k == 1) {
        // Match the general path's tie break by choosing the largest index.
        int best = 0;
        for (int i = 1; i < count; ++i) {
            if (values[i] >= values[best]) {
                best = i;
            }
        }
        ranked = {{values[best]}, {best}};
    } else {
        ranked = topk(vector<float>(values, values + count), k);
    }

    _Topks result;
    result.scores = std::move(ranked.scores);
    result.indexs = std::move(ranked.indexs);
    result.clses.assign(result.indexs.size(), 0);
    for (const int index : result.indexs) {
        result.ys.push_back(static_cast<float>(index / scores.w));
        result.xs.push_back(static_cast<float>(index % scores.w));
    }
    return result;
}

ncnn::Mat equal(ncnn::Mat heatmap, const ncnn::Mat& pooled) {
    if (!is_heatmap(heatmap) || !is_heatmap(pooled) ||
        heatmap.w != pooled.w || heatmap.h != pooled.h) {
        throw std::invalid_argument("Model heatmap and pooled heatmap must have matching dimensions");
    }
    // The extracted heatmap is only used for peak selection after this call.
    for (int y = 0; y < heatmap.h; ++y) {
        float* scores = heatmap.row(y);
        const float* maxima = pooled.row(y);
        for (int x = 0; x < heatmap.w; ++x) {
            if (!std::isfinite(scores[x]) || scores[x] != maxima[x]) {
                scores[x] = 0;
            }
        }
    }
    return heatmap;
}

bboxs bbox_decode(const _Topks& peaks, const ncnn::Mat& corners, const ncnn::Mat& offsets,
                  const vector<int>& angles, const vector<int>& types, float scale) {
    const auto count = peaks.indexs.size();
    if (count == 0) {
        return {};
    }
    if (peaks.scores.size() != count || peaks.xs.size() != count || peaks.ys.size() != count ||
        angles.size() != count || types.size() != count) {
        throw std::invalid_argument("Each peak must have coordinates, a score, an angle and a type");
    }
    require_head(corners, 8, corners.h, "corners");
    if (!offsets.empty()) {
        require_head(offsets, 2, corners.h, "offsets");
    }

    bboxs boxes;
    boxes.ids = peaks.indexs;
    boxes.detections.reserve(count);
    const float pixel_scale = model_stride * scale;
    for (std::size_t i = 0; i < count; ++i) {
        const int index = peaks.indexs[i];
        if (index < 0 || index >= corners.h) {
            throw std::out_of_range("Peak index is outside the corners tensor");
        }
        const float* corner = corners.row(index);
        const float* offset = offsets.empty() ? nullptr : offsets.row(index);
        const float x = peaks.xs[i] + (offset ? offset[0] : 0.5f);
        const float y = peaks.ys[i] + (offset ? offset[1] : 0.5f);

        vector<float> box;
        box.reserve(11);
        for (int j = 0; j < 8; j += 2) {
            box.push_back((x - corner[j]) * pixel_scale);
            box.push_back((y - corner[j + 1]) * pixel_scale);
        }
        box.push_back(peaks.scores[i]);
        box.push_back(static_cast<float>(angles[i]));
        box.push_back(static_cast<float>(types[i]));
        boxes.detections.push_back(std::move(box));
    }
    return boxes;
}

vector<int> max_2dim(const vector<vector<float>>& classes) {
    vector<int> result;
    result.reserve(classes.size());
    for (const auto& row : classes) {
        if (row.empty() || !std::all_of(row.begin(), row.end(), [](float value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("Classification row has missing or invalid scores");
        }
        result.push_back(static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin()));
    }
    return result;
}

vector<int> decode_by_ind(const ncnn::Mat& heat_mat, const vector<int>& inds) {
    return max_2dim(gather_feat(heat_mat, inds));
}

float distance(float x1, float y1, float x2, float y2) {
    return std::hypot(x1 - x2, y1 - y2);
}

cv::Mat crop_image(const cv::Mat& image, const vector<cv::Point2f>& points) {
    if (image.empty() || points.size() != 4 ||
        !std::all_of(points.begin(), points.end(), [](const cv::Point2f& point) {
            return std::isfinite(point.x) && std::isfinite(point.y);
        }) ||
        !cv::isContourConvex(points) || cv::contourArea(points) < 1) {
        throw std::invalid_argument("Crop requires a nonempty image and a valid quadrilateral");
    }

    const auto& p0 = points[0];
    const auto& p1 = points[1];
    const auto& p2 = points[2];
    const auto& p3 = points[3];
    const double width = cv::norm(((p0 + p3) - (p1 + p2)) * 0.5f);
    const double height = cv::norm(((p0 + p1) - (p2 + p3)) * 0.5f);
    const double max_side = std::min(std::hypot(static_cast<double>(image.cols), image.rows),
                                     static_cast<double>(std::numeric_limits<int>::max()));
    if (!std::isfinite(width) || !std::isfinite(height) ||
        width < 1 || height < 1 || width > max_side || height > max_side) {
        throw std::invalid_argument("Crop dimensions are outside the valid range");
    }

    const float image_width = static_cast<float>(image.cols);
    const float image_height = static_cast<float>(image.rows);
    const vector<cv::Point2f> image_bounds = {{0, 0}, {image_width, 0},
                                              {image_width, image_height}, {0, image_height}};
    vector<cv::Point2f> overlap;
    if (cv::intersectConvexConvex(points, image_bounds, overlap) <= 0) {
        throw std::invalid_argument("Crop is outside the source image");
    }

    const auto crop_width = static_cast<float>(width);
    const auto crop_height = static_cast<float>(height);
    const cv::Point2f destination[] = {{0, 0}, {crop_width, 0},
                                       {crop_width, crop_height}, {0, crop_height}};
    const auto transform = cv::getPerspectiveTransform(points.data(), destination);
    cv::Mat result;
    cv::warpPerspective(image, result, transform,
                        cv::Size(static_cast<int>(width), static_cast<int>(height)));
    return result;
}

cv::Mat crop_image(const cv::Mat& image, const vector<float>& box) {
    return crop_image(image, corners_from_box(box));
}
