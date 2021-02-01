/*
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: PlateNumPostProcessMethod.h
 * @Brief: declaration of PlateNumPostProcessMethod
 * @Author: shiyu.fu
 * @Email: shiyu.fu@horizon.ai
 * @Date: Mon Jan 04 2021 11:48:10
 */

#ifndef INCLUDE_PLATENUMPOSTPROCESSMETHOD_PLATENUMPOSTPROCESSMETHOD_H_
#define INCLUDE_PLATENUMPOSTPROCESSMETHOD_PLATENUMPOSTPROCESSMETHOD_H_

#include <memory>
#include <string>
#include <vector>
#include "DnnAsyncData.h"
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "bpu_predict/bpu_parse_utils_extension.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class PlateNumPostProcessMethod : public DnnPostProcessMethod {
 public:
  PlateNumPostProcessMethod() {}
  virtual ~PlateNumPostProcessMethod() {}

  int Init(const std::string &cfg_path) override;

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  int ParseDnnResult(DnnAsyncData &dnn_result,
                     std::vector<BaseDataPtr> &frame_result) override;

 private:
  void ParsePlateNum(BPU_MODEL_S bpu_model, std::vector<int> &valid_box,
                     std::vector<std::vector<float>> &features,
                     std::shared_ptr<BaseDataVector> results);

 private:
  uint32_t output_size_;
};

}  // namespace xstream
#endif  // INCLUDE_PLATENUMPOSTPROCESSMETHOD_PLATENUMPOSTPROCESSMETHOD_H_
