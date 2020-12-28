/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: DnnPostProcessMethod.h
 * @Brief: declaration of the DnnPostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-11-24 13:35:19
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-11-24 14:04:23
 */

#ifndef INCLUDE_DNNPOSTPROCESSMETHOD_DNNPOSTPROCESSMETHOD_H_
#define INCLUDE_DNNPOSTPROCESSMETHOD_DNNPOSTPROCESSMETHOD_H_

#include <mutex>
#include <string>
#include <vector>
#include "json/json.h"
#include "Config.h"
#include "hobotxstream/method.h"
#include "DnnAsyncData.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class DnnPostProcessMethod : public Method {
 public:
  DnnPostProcessMethod() {}
  virtual ~DnnPostProcessMethod() {}

  int Init(const std::string &cfg_path) override;
  void Finalize() override {}

  // 主逻辑，完全复用，派生类不需要再实现DoProcess
  std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<xstream::InputParamPtr> &param) override;

 public:
  static std::mutex init_mutex_;
  Config config_;

  // 释放InputTensor/OutputTensor
  void FreeTensor(std::vector<BPU_TENSOR_S> &tensors);

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式;不需考虑tensor的释放
  // IN: dnn_result. OUT: frame_result
  virtual int ParseDnnResult(DnnAsyncData &dnn_result,
                             std::vector<BaseDataPtr> &frame_result) {
    return -1;
  }
};
}  // namespace xstream
#endif  // INCLUDE_DNNPOSTPROCESSMETHOD_DNNPOSTPROCESSMETHOD_H_
