/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: MattingTrimapFreePostProcessMethod.cpp
 * @Brief: definition of the MattingTrimapFreePostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-15 20:06:05
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-15 20:43:08
 */

#include "MattingTrimapFreePostProcessMethod/MattingTrimapFreePostProcessMethod.h"
#include <string>
#include <vector>
#include <fstream>
#include "hobotxstream/profiler.h"
#include "dnn_util.h"
#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "hobotlog/hobotlog.hpp"

namespace xstream {

int MattingTrimapFreePostProcessMethod::ParseDnnResult(
    DnnAsyncData &dnn_result,
    std::vector<BaseDataPtr> &frame_result) {
  LOGD << "MattingTrimapFreePostProcessMethod ParseDnnResult";
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
    // parse valid output_tensor: 3层 1x512x512x1 int8 NHWC
    HOBOT_CHECK(input_tensor.size() == 1);  // one input layer
    HOBOT_CHECK(output_tensor.size() == 3);  // three output layer
    std::vector<std::vector<float>> out_datas(3);

    // 需要移位(与模型有关, 和算法确认即可)
    for (size_t i = 0; i < 3; i++) {
      HB_SYS_flushMemCache(&(output_tensor[i].data),
                           HB_SYS_MEM_CACHE_INVALIDATE);
      int model_out_size =
          dnn_result.dnn_model->bpu_model.outputs[i].shape.d[0] *
          dnn_result.dnn_model->bpu_model.outputs[i].shape.d[1] *
          dnn_result.dnn_model->bpu_model.outputs[i].shape.d[2] *
          dnn_result.dnn_model->bpu_model.outputs[i].shape.d[3];
      std::vector<float> one_layer_data(model_out_size);
      RUN_PROCESS_TIME_PROFILER("Convert_Float");
      ConvertOutputToFloat(
          output_tensor[i].data.virAddr, one_layer_data.data(),
          dnn_result.dnn_model->bpu_model, i);
      out_datas[i] = one_layer_data;
    }

    // postprocess
    cv::Mat fg(512, 512, CV_32FC1, out_datas[0].data());
    cv::Mat bg(512, 512, CV_32FC1, out_datas[1].data());
    cv::Mat fusion_mask(512, 512, CV_32FC1, out_datas[2].data());
    cv::Mat weighted_fg, weighted_bg, matting_pred;
    cv::multiply(fg, fusion_mask, weighted_fg);
    cv::multiply(1.0 - bg, 1.0 - fusion_mask, weighted_bg);
    matting_pred = weighted_fg + weighted_bg;

    auto segmentation = std::make_shared<XStreamData<
                            hobot::vision::Segmentation>>();
    hobot::vision::Segmentation matting;
    matting.values = std::vector<float>(matting_pred.reshape(1, 1));
    segmentation->value = std::move(matting);
    matting_result->datas_.push_back(segmentation);
  }
  return 0;
}

}  // namespace xstream
