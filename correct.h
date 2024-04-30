//
// Created by 钟顺民 on 2024/4/30.
// email 734001892@qq.com z734001892@gmail.com
// 单据、证件矫正
//

#include <iostream>
#include "net.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
using namespace ncnn;

const int target_size = 768;


struct bboxs{
    vector<vector<float>> detections;
    std::vector<int> ids;
};


struct Topks{
    std::vector<float> scores;
    std::vector<int> indexs;
};


struct _Topks{
    std::vector<float> scores;
    std::vector<int> indexs;
    std::vector<int> clses;
    std::vector<float> ys;
    std::vector<float> xs;
};

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b);
cv::Mat getAffineTransform(const cv::Point2f& center, float scale);
cv::Point2f transformPreds( cv::Point2f pt,  cv::Point2f center, float scale);
vector<cv::Point2f> bbox_post_process(vector<float> bbox_top1,cv::Point2f center, float scale);
std::vector<vector<float>> gather_feat(ncnn::Mat heat_mat,vector<int> inds);

Topks topk(std::vector<float>& cls_scores, int topk);
_Topks _topk(ncnn::Mat scores,int k);
ncnn::Mat equal(ncnn::Mat hitmap, ncnn::Mat hitmap_pool);
bboxs bbox_decode(_Topks _topks,ncnn::Mat wh_mat,ncnn::Mat reg,vector<int> angle,vector<int> ftype,float scale);
vector<int> max_2dim(vector<vector<float>> clses);
vector<int> decode_by_ind(ncnn::Mat heat_mat,vector<int> inds);
float distance(float x1, float y1, float x2,float y2);
cv::Mat crop_image(cv::Mat rgb,vector<cv::Point2f> bbox);
cv::Mat crop_image(cv::Mat rgb,vector<float> bbox);
