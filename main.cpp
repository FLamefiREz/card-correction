//
// Created by 钟顺民 on 2024/4/30.
// email 734001892@qq.com z734001892@gmail.com
// 单据、证件矫正
//

#include <iostream>
#include "net.h"
#include "opencv2/opencv.hpp"
#include <vector>

#include "correct.h"

using namespace cv;
using namespace std;
using namespace ncnn;


cv::Mat correction(cv::Mat rgb){
    int width = rgb.cols;
    int height = rgb.rows;

    int new_size = std::max(width, height);

    cv::Mat square_img = cv::Mat::zeros(cv::Size(new_size, new_size), rgb.type());

    int x_offset = (new_size - width) / 2;
    int y_offset = (new_size - height) / 2;

    cv::Mat roi(square_img, cv::Rect(x_offset, y_offset, width, height));
    rgb.copyTo(roi);

    float scale = (float)new_size/target_size;

    const float meanValues[3] = {0.408 * 255, 0.456 * 255, 0.470 * 255};
    const float normValues[3] = {1.0 / 0.289 / 255.0, 1.0 / 0.274 / 255.0,
                                 1.0 / 0.278 / 255.0};

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(square_img.data,ncnn::Mat::PIXEL_RGB, new_size, new_size, target_size, target_size);
    input.substract_mean_normalize(meanValues, normValues);
    Net model;
    model.load_param("./model/correction.param");
    model.load_model("./model/correction.bin");

    Extractor ex = model.create_extractor();
    ex.input("in0",input);
    ncnn::Mat heatmap,heatmap_pool,wh,angle_cls,ftype_cls,reg;

    ex.extract("/Sigmoid_output_0", heatmap);
    ex.extract("out0", heatmap_pool);
    ex.extract("out1", wh);
    ex.extract("out2", angle_cls);
    ex.extract("out3", ftype_cls);
    ex.extract("out4", reg);
    ncnn::Mat hitmap_nms = equal(heatmap,heatmap_pool);
    _Topks _topks = _topk(hitmap_nms,10);
    vector<int> ins = _topks.indexs;
    vector<int> angle = decode_by_ind(angle_cls,ins);
    vector<int> ftype = decode_by_ind(ftype_cls,ins);
    bboxs bboxs = bbox_decode(_topks,wh,reg,angle,ftype,scale);
    vector<float> bbox_top1 = bboxs.detections[0];
    cv::Mat outputImage;
    if(bbox_top1[8]<0.3){
        return rgb;
    }
    for (int i = 0; i < 4; ++i) {
        fprintf(stdout, "x%d = %f, y%d = %f\n", i, bbox_top1[2*i],i,bbox_top1[2*i+1]);
    }
    outputImage = crop_image(square_img,bbox_top1);
    if (angle[0] == 1)
    {cv::rotate(outputImage, outputImage,2);}
    if (angle[0] == 2)
    {cv::rotate(outputImage, outputImage,1);}
    if (angle[0] == 3)
    {cv::rotate(outputImage, outputImage,0);}

    return outputImage;
}

int main() {
    cv::Mat rgb = cv::imread("./example/1.jpg");
    cv::imshow("src",rgb);
    cv::Mat outputImage = correction(rgb);
    cv::imshow("result",outputImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::imwrite("result.jpg",outputImage);
    return 0;
}
