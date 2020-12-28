/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: DnnPostProcessMethod.cpp
 * @Brief: definition of the DnnPostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-11-24 13:56:05
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-11-24 16:23:08
 */

#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include <fstream>
#include <string>
#include <vector>
#include "Config.h"
#include "hobotxstream/profiler.h"
#include "hobotlog/hobotlog.hpp"
#include "DnnAsyncData.h"

namespace xstream {

std::mutex DnnPostProcessMethod::init_mutex_;
int32_t DnnPostProcessMethod::Init(const std::string &cfg_path) {
  LOGI << "DnnPostProcessMethod Init";
  // 0. parse config_
  Json::Value config;
  std::ifstream infile(cfg_path.c_str());
  if (!infile.good()) {
    LOGW << "DnnPostProcessMethod error config file path: " << cfg_path;
  } else {
    infile >> config;
  }
  config_ = Config(config);
  return 0;
}

void DnnPostProcessMethod::FreeTensor(std::vector<BPU_TENSOR_S> &tensors) {
  for (size_t i = 0; i < tensors.size(); i++) {
    BPU_TENSOR_S &tensor = tensors[i];
    switch (tensor.data_type) {
      case BPU_TYPE_IMG_Y:
      case BPU_TYPE_IMG_YUV_NV12:
      case BPU_TYPE_IMG_YUV444:
      case BPU_TYPE_IMG_RGB:
      case BPU_TYPE_IMG_BGR:
      case BPU_TYPE_IMG_BGRP:
      case BPU_TYPE_IMG_RGBP:
      case BPU_TYPE_TENSOR_U8:
      case BPU_TYPE_TENSOR_S8:
      case BPU_TYPE_TENSOR_F32:
      case BPU_TYPE_TENSOR_S32:
      case BPU_TYPE_TENSOR_U32:
        HB_SYS_bpuMemFree(&tensor.data);
        break;
      case BPU_TYPE_IMG_NV12_SEPARATE:
        HB_SYS_bpuMemFree(&tensor.data);
        HB_SYS_bpuMemFree(&tensor.data_ext);
        break;
      default:
        HOBOT_CHECK(0) << "not support data_type: " << tensor.data_type;
        break;
    }
  }
}

std::vector<std::vector<BaseDataPtr>> DnnPostProcessMethod::DoProcess(
    const std::vector<std::vector<BaseDataPtr>> &input,
    const std::vector<xstream::InputParamPtr> &param) {
  LOGD << "DnnPostProcessMethod DoProcess";
  HOBOT_CHECK(!input.empty());
  RUN_PROCESS_TIME_PROFILER("DnnPostProcessMethod_DoProcess");
  std::vector<std::vector<BaseDataPtr>> output;
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    const std::vector<BaseDataPtr> &frame_input = input[i];
    std::vector<BaseDataPtr> &frame_output = output[i];

    // DnnPostProcessMethod的输入只有一个，输入数据就是DnnAsyncData
    HOBOT_CHECK(frame_input.size() == 1)  << "only support DnnAsyncData";
    auto dnn_async_data =
        std::static_pointer_cast<DnnAsyncData>(frame_input[0]);

    // 调用HB_BPU_waitModelDone接口，等待bpu异步任务完成
    if (!dnn_async_data->dnn_is_sync) {
      for (size_t i = 0; i < dnn_async_data->task_handle.size(); ++i) {
        BPU_TASK_HANDLE &task_handle = dnn_async_data->task_handle[i];
        if (task_handle == nullptr) continue;
        HB_BPU_waitModelDone(&task_handle);
        HB_BPU_releaseTask(&task_handle);
      }
    }

    // 调用派生类的接口，完成模型后处理，将结果转换成Method的输出格式
    {
      RUN_PROCESS_TIME_PROFILER("ParseDnnResult");
      int ret = ParseDnnResult(*dnn_async_data, frame_output);
      if (ret != 0) {
        LOGE << "ParseDnnResult failed";
      }
    }
    // 释放输入与输出Tensor
    for (size_t i = 0; i < dnn_async_data->input_tensors.size(); ++i) {
      if (!dnn_async_data->input_tensors[i].empty()) {
        FreeTensor(dnn_async_data->input_tensors[i]);
      }
    }
    for (size_t i = 0; i < dnn_async_data->output_tensors.size(); ++i) {
      if (!dnn_async_data->output_tensors[i].empty()) {
        FreeTensor(dnn_async_data->output_tensors[i]);
      }
    }
  }
  return output;
}

}  // namespace xstream
