/*
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: VehicleColorPostProcessMethod.h
 * @Brief: declaration of VehicleColorPostProcessMethod
 * @Author: shiyu.fu
 * @Email: shiyu.fu@horizon.ai
 * @Date: Mon Jan 04 2021 10:56:29
 */

#ifndef INCLUDE_VEHICLECOLORPOSTPROCESSMETHOD_VEHICLECOLORPOSTPROCESSMETHOD_H_
#define INCLUDE_VEHICLECOLORPOSTPROCESSMETHOD_VEHICLECOLORPOSTPROCESSMETHOD_H_

#include <memory>
#include <string>
#include <vector>
#include "DnnAsyncData.h"
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "bpu_predict/bpu_parse_utils_extension.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class VehicleColorPostProcessMethod : public DnnPostProcessMethod {
 public:
  VehicleColorPostProcessMethod() {}
  virtual ~VehicleColorPostProcessMethod() {}

  int Init(const std::string &cfg_path) override;

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  int ParseDnnResult(DnnAsyncData &dnn_result,
                     std::vector<BaseDataPtr> &frame_result) override;

 private:
  void ParseVehicleColor(BPU_MODEL_S bpu_model, std::vector<int> &valid_box,
                         BPU_TENSOR_S output_tensor,
                         std::shared_ptr<BaseDataVector> results);

 private:
  uint32_t output_size_;
};

}  // namespace xstream
#endif  // INCLUDE_VEHICLECOLORPOSTPROCESSMETHOD_VEHICLECOLORPOSTPROCESSMETHOD_H_   // NOLINT
