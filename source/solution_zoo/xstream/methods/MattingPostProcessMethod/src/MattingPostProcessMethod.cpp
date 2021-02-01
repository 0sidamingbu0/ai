/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: MattingPostProcessMethod.cpp
 * @Brief: definition of the MattingPostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-11-26 13:36:05
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2029-11-26 18:23:08
 */

#include "MattingPostProcessMethod/MattingPostProcessMethod.h"
#include <string>
#include <vector>
#include <fstream>
#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "hobotlog/hobotlog.hpp"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

int MattingPostProcessMethod::Init(const std::string &cfg_path) {
  DnnPostProcessMethod::Init(cfg_path);
  threshold_ = config_.GetIntValue("threshold", threshold_);
  return 0;
}

int MattingPostProcessMethod::ParseDnnResult(
    DnnAsyncData &dnn_result,
    std::vector<BaseDataPtr> &frame_result) {
  LOGD << "MattingPostProcessMethod ParseDnnResult";
  frame_result.resize(1);  // matting result, represented as Segmentation
  auto matting_result = std::make_shared<xstream::BaseDataVector>();
  frame_result[0] = matting_result;

  auto &input_tensors = dnn_result.input_tensors;
  auto &output_tensors = dnn_result.output_tensors;
  for (size_t i = 0; i < output_tensors.size(); i++) {
    auto &input_tensor = input_tensors[i];
    auto &output_tensor = output_tensors[i];
    if (output_tensor.size() == 0) {
      auto segmentation = std::make_shared<XStreamData<
                            hobot::vision::Segmentation>>();
      segmentation->state_ = DataState::INVALID;
      matting_result->datas_.push_back(segmentation);
      continue;
    }
    // parse valid output_tensor
    HOBOT_CHECK(input_tensor.size() == 1);  // one input layer
    HOBOT_CHECK(output_tensor.size() == 1);  // one output layer
    static float bpu_result[256*256];
    HB_SYS_flushMemCache(&(output_tensor[0].data),
                         HB_SYS_MEM_CACHE_INVALIDATE);
    memcpy(bpu_result, output_tensor[0].data.virAddr, 256*256*sizeof(float));

    // postprocess
    float *trimap_float = reinterpret_cast<float *>(
        input_tensor[0].data.virAddr);
    int size = 256 * 256;
    trimap_float = trimap_float + 3 * size;
    cv::Mat trimap_float_mat(256, 256, CV_32FC1, trimap_float);
    cv::Mat trimap_uint_mat;
    trimap_float_mat.convertTo(trimap_uint_mat, CV_8UC1);
    uchar* trimap = trimap_uint_mat.data;
    for (int i = 0; i < size; i++) {
      if (trimap[i] == 2) {
        bpu_result[i] = 1;
      } else if (trimap[i] == 0) {
        bpu_result[i] = 0;
      }
    }

    auto segmentation = std::make_shared<XStreamData<
                            hobot::vision::Segmentation>>();
    hobot::vision::Segmentation matting;
    matting.values = std::vector<float>(bpu_result, bpu_result+256*256);
    segmentation->value = std::move(matting);
    matting_result->datas_.push_back(segmentation);
  }
  return 0;
}

}  // namespace xstream
