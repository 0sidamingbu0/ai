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

int MattingTrimapFreePostProcessMethod::Init(const std::string &cfg_path) {
  DnnPostProcessMethod::Init(cfg_path);
  matting_thresh_ = config_.GetFloatValue("matting_thresh", matting_thresh_);
  return 0;
}

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
    // 定点转浮点耗时较长，若模型3层输出的移位值相等，可以先取值，再移位
    static bool need_convert_float = false;
    static std::once_flag flag;
    static int shift_value;
    auto &bpu_model = dnn_result.dnn_model->bpu_model;
    std::call_once(flag, [&bpu_model]() {
      int layer = bpu_model.output_num;
      bool shift_init = false;
      for (int i = 0; i < layer; i++) {
        auto shift = bpu_model.outputs[i].shifts;
        int h_idx, w_idx, c_idx;
        HB_BPU_getHWCIndex(bpu_model.outputs[i].data_type,
                           &bpu_model.outputs[i].shape.layout,
                           &h_idx, &w_idx, &c_idx);
        int channel = bpu_model.outputs[i].shape.d[c_idx];
        HOBOT_CHECK(channel == 1) << "model output channel should be 1";
        if (!shift_init) {
          shift_value = shift[0];
          shift_init = true;
        } else {
          if (shift_value != shift[0]) {
            need_convert_float = true;
            break;
          }
        }
      }
    });
    // 刷新flush
    for (size_t i = 0; i < 3; i++) {
      HB_SYS_flushMemCache(&(output_tensor[i].data),
                           HB_SYS_MEM_CACHE_INVALIDATE);
    }
    int model_output_height, model_output_width;
    HB_BPU_getHW(dnn_result.dnn_model->bpu_model.outputs[0].data_type,
                 &dnn_result.dnn_model->bpu_model.outputs[0].shape,
                 &model_output_height, &model_output_width);

    if (need_convert_float) {
      for (size_t i = 0; i < 3; i++) {
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
    } else {
      // 取bpu定点数据,存到Mat
      std::vector<cv::Mat> outs(3);
      std::vector<std::vector<cv::Mat>> split_outs(3);
      for (int i = 0; i < 3; i++) {
        outs[i] = cv::Mat(model_output_height, model_output_width,
                          CV_8SC4, output_tensor[i].data.virAddr);
        // 有效数据为第0通道
        cv::split(outs[i], split_outs[i]);
        // split_outs[i][0]是有效数据
        split_outs[i][0].convertTo(split_outs[i][0], CV_32FC1);  // 转为浮点
        split_outs[i][0] = split_outs[i][0] / (1 << shift_value);
        out_datas[i] = split_outs[i][0].reshape(1, 1);  // 存到vector
      }
    }

    // postprocess
    cv::Mat weighted_fg, weighted_bg, matting_pred;
    cv::Mat fg(model_output_height, model_output_width,
               CV_32FC1, out_datas[0].data());
    cv::Mat bg(model_output_height, model_output_width,
               CV_32FC1, out_datas[1].data());
    cv::Mat fusion_mask(model_output_height, model_output_width,
                        CV_32FC1, out_datas[2].data());
    cv::multiply(fg, fusion_mask, weighted_fg);
    cv::multiply(1.0 - bg, 1.0 - fusion_mask, weighted_bg);
    cv::Mat matting_res;
    matting_res = weighted_fg + weighted_bg;
    // 元素值>=matting_thresh_ 得到255，否则0
    // cv::compare(matting_res, matting_thresh_, matting_pred, cv::CMP_GE);
    matting_res *= 255;

    auto segmentation = std::make_shared<XStreamData<
                            hobot::vision::Segmentation>>();
    hobot::vision::Segmentation matting;
    matting.values = std::vector<float>(matting_res.reshape(1, 1));
    matting.width = model_output_width;
    matting.height = model_output_height;
    segmentation->value = std::move(matting);
    matting_result->datas_.push_back(segmentation);
  }
  return 0;
}

}  // namespace xstream
