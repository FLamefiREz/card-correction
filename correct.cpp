//
// Created by 钟顺民 on 2024/4/30.
// email 734001892@qq.com z734001892@gmail.com
// 单据、证件矫正
//

#include "correct.h"

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f direct = b - a;
    return {a.y - direct.y, a.x + direct.x};
}


cv::Mat getAffineTransform(const cv::Point2f& center, float scale) {
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];
    cv::Point2f srcDir = cv::Point2f(0, -0.5 * scale);
    cv::Point2f dstDir(0, -0.5 * 192);
    srcTri[0] = center;
    srcTri[1] = center + srcDir;
    dstTri[0] = cv::Point2f(0.5 * 192, 0.5 * 192);
    dstTri[1] = cv::Point2f(0.5 * 192, 0.5 * 192) + dstDir;
    srcTri[2] = get3rdPoint(srcTri[0], srcTri[1]);
    dstTri[2] = get3rdPoint(dstTri[0], dstTri[1]);
    cv::Mat transform;

    transform = cv::getAffineTransform(dstTri, srcTri);

    return transform;
}

cv::Point2f transformPreds( cv::Point2f pt,  cv::Point2f center, float scale) {
    cv::Mat trans = getAffineTransform(center, scale);
    cv::Point2f targetCoords = pt;
    cv::Mat ptMat(cv::Size(1, 3),6);
    ptMat.at<float>(0, 0) = pt.x;
    ptMat.at<float>(1, 0) = pt.y;
    ptMat.at<float>(2, 0) = 1;
    cv::Mat newPtMat = trans * ptMat;

    targetCoords.x = newPtMat.at<float>(0, 0);
    targetCoords.y = newPtMat.at<float>(1, 0);
    return targetCoords;
}

vector<cv::Point2f> bbox_post_process(vector<float> bbox_top1,cv::Point2f center, float scale){
    vector<cv::Point2f> points;
    for (int i = 0; i < 4; ++i) {
        cv::Point2f point = transformPreds(cv::Point2f(bbox_top1[2*i], bbox_top1[2*i+1]),center, scale);
        points.push_back(point);
    }
    return points;
}

std::vector<vector<float>> gather_feat(ncnn::Mat heat_mat,vector<int> inds){
    std::vector<vector<float>> heat;
    std::vector<float> heat_;
    for (int i = 0; i < inds.size(); ++i) {
        for (int j = 0; j < heat_mat.w; ++j) {
            heat_.emplace_back(heat_mat.row(inds[i])[j]);
        }
        heat.emplace_back(heat_);
    }
    return heat;
}


Topks topk(std::vector<float>& cls_scores, int topk)
{
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    std::vector<float> scores;
    std::vector<int> indexs;
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        scores.emplace_back(score);
        indexs.emplace_back(index);
    }
    Topks topks;
    topks.scores=scores;
    topks.indexs=indexs;
    return topks;
}

_Topks _topk(ncnn::Mat scores,int k){
    int height = scores.h;
    int width = scores.w;
    _Topks _topks;
    std::vector<float> cls_scores;
    for (int i = 0; i < scores.w; ++i) {
        for (int j = 0; j < scores.h; ++j) {
            cls_scores.emplace_back(scores[i*192+j]);
        }
    }
    Topks topks = topk(cls_scores,k);
    std::vector<float> topk_scores = topks.scores;
    std::vector<int> topk_inds = topks.indexs;
    std::vector<float> topk_ys;
    std::vector<float> topk_xs;
    std::vector<int> topk_clses;
    for (int i = 0; i < topk_inds.size(); ++i) {
        topk_inds[i] =topk_inds[i] % (height * width);
        topk_ys.emplace_back(topk_inds[i] / width *1.0f);
        topk_xs.emplace_back(topk_inds[i] % width *1.0f);
        topk_clses.emplace_back(0);
    }
    Topks topks_ = topk(topk_scores,k);
    std::vector<float> topk_score = topks_.scores;
    std::vector<int> topk_ind = topks_.indexs;

    _topks.indexs = topk_inds;
    _topks.scores = topk_score;
    _topks.ys = topk_ys;
    _topks.xs = topk_xs;
    _topks.clses = topk_clses;
    return _topks;
}


ncnn::Mat equal(ncnn::Mat hitmap, ncnn::Mat hitmap_pool){
    for (int i = 0; i < hitmap.w; ++i) {
        for (int j = 0; j < hitmap.h; ++j) {
            if(hitmap[i*192+j]==hitmap_pool[i*192+j]){
                hitmap[i*192+j]=hitmap[i*192+j]*1.0f;
            }
            else{
                hitmap[i*192+j]=hitmap[i*192+j]*0.0f;
            }
        }
    }
    return hitmap;
}

bboxs bbox_decode(_Topks _topks,ncnn::Mat wh_mat,ncnn::Mat reg,vector<int> angle,vector<int> ftype,float scale){
    bboxs bboxs_;
    vector<float> xs = _topks.xs;
    vector<float> ys = _topks.ys;
    vector<vector<float>> bboxs;
    vector<vector<float>> wh;

    if(!reg.empty()){
        for (int i = 0; i < _topks.indexs.size(); ++i) {
            vector<float> wh_;
            xs[i] = xs[i] + reg[_topks.indexs[i]*2];
            ys[i] = ys[i] + reg[_topks.indexs[i]*2+1];
            for (int j = 0; j < wh_mat.w; ++j) {
                wh_.emplace_back(wh_mat[_topks.indexs[i]*8+j]);
            }
            wh.emplace_back(wh_);
        }
    }else{
        for (int i = 0; i < _topks.indexs.size(); ++i) {
            vector<float> wh_;
            xs[i] = xs[i] + 0.5f;
            ys[i] = ys[i] + 0.5f;
            for (int j = 0; j < wh_mat.w; ++j) {
                wh_.emplace_back(wh_mat[_topks.indexs[i]*8+j]);
            }
            wh.emplace_back(wh_);
        }
    }

    for (int i = 0; i < ys.size(); ++i) {
        vector<float> bbox;
        for (int j = 0; j < wh_mat.w; ++j) {
            if(j%2){
                bbox.emplace_back((ys[i]-wh[i][j])*4*scale);
            }else{
                bbox.emplace_back((xs[i]-wh[i][j])*4*scale);
            }
        }
        bbox.emplace_back(_topks.scores[i]);
        bbox.emplace_back(angle[i]);
        bbox.emplace_back(ftype[i]);
        bboxs.emplace_back(bbox);
    }

    bboxs_.detections =bboxs;
    bboxs_.ids=_topks.indexs;
    return bboxs_;
}

vector<int> max_2dim(vector<vector<float>> clses){
    vector<int> result;
    for (int i = 0; i <clses.size(); ++i) {
        vector<float> cls = clses[i];
        int index=0;
        float max=0.f;
        for (int j = 0; j < cls.size(); ++j) {
            if(cls[j]>=max){
                index = j;
                max = cls[j];
            }
        }
        result.push_back(index);
    }
    return result;
}

vector<int> decode_by_ind(ncnn::Mat heat_mat,vector<int> inds){
    vector<vector<float>> clses = gather_feat(heat_mat,inds);
    vector<int> result = max_2dim(clses);
    return result;
}

float distance(float x1, float y1, float x2,float y2){
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

cv::Mat crop_image(cv::Mat rgb,vector<cv::Point2f> bbox){
    float x0 = bbox[0].x;
    float y0 = bbox[0].y;
    float x1 = bbox[1].x;
    float y1 = bbox[1].y;
    float x2 = bbox[2].x;
    float y2 = bbox[2].y;
    float x3 = bbox[3].x;
    float y3 = bbox[3].y;
    float img_width = distance((x0 + x3) / 2, (y0 + y3) / 2, (x1 + x2) / 2,
                               (y1 + y2) / 2);
    float img_height = distance((x0 + x1) / 2, (y0 + y1) / 2, (x2 + x3) / 2,
                                (y2 + y3) / 2);
    std::vector<cv::Point2f> srcPoints;
    srcPoints.emplace_back(x0, y0);
    srcPoints.emplace_back(x1, y1);
    srcPoints.emplace_back(x2, y2);
    srcPoints.emplace_back(x3, y3);

    std::vector<cv::Point2f> dstPoints;
    dstPoints.emplace_back(0, 0);
    dstPoints.emplace_back(img_width, 0);
    dstPoints.emplace_back(img_width, img_height);
    dstPoints.emplace_back(0, img_height);

    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    cv::Mat outputImage;
    cv::warpPerspective(rgb, outputImage, perspectiveMatrix, cv::Size(img_width, img_height));
    return outputImage;
}

cv::Mat crop_image(cv::Mat rgb,vector<float> bbox){
    float x0 = bbox[0];
    float y0 = bbox[1];
    float x1 = bbox[2];
    float y1 = bbox[3];
    float x2 = bbox[4];
    float y2 = bbox[5];
    float x3 = bbox[6];
    float y3 = bbox[7];
    float img_width = distance((x0 + x3) / 2, (y0 + y3) / 2, (x1 + x2) / 2,
                               (y1 + y2) / 2);
    float img_height = distance((x0 + x1) / 2, (y0 + y1) / 2, (x2 + x3) / 2,
                                (y2 + y3) / 2);
    std::vector<cv::Point2f> srcPoints;
    srcPoints.emplace_back(x0, y0);
    srcPoints.emplace_back(x1, y1);
    srcPoints.emplace_back(x2, y2);
    srcPoints.emplace_back(x3, y3);
    // 定义目标图像上的四个点
    std::vector<cv::Point2f> dstPoints;
    dstPoints.emplace_back(0, 0);
    dstPoints.emplace_back(img_width, 0);
    dstPoints.emplace_back(img_width, img_height);
    dstPoints.emplace_back(0, img_height);
    // 计算透视变换矩阵
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
    // 进行透视变换
    cv::Mat outputImage;
    cv::warpPerspective(rgb, outputImage, perspectiveMatrix, cv::Size(img_width, img_height));
    return outputImage;
}