/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: mobilenetv2_predict_method.h
 * @Brief: declaration of the Mobilenetv2PredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-24 17:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-24 20:21:33
 */

#ifndef MOBILENETV2PREDICTMETHOD_MOBILENETV2PREDICTMETHOD_H_
#define MOBILENETV2PREDICTMETHOD_MOBILENETV2PREDICTMETHOD_H_

#include <string>
#include <vector>
#include "DnnPredictMethod/DnnPredictMethod.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class Mobilenetv2PredictMethod : public DnnPredictMethod {
 public:
  Mobilenetv2PredictMethod() {}
  virtual ~Mobilenetv2PredictMethod() {}

  int Init(const std::string &cfg_path) override;

  // 派生类需要实现
  // PrepareInputData内部需要根据一帧图像目标数量，多次调用AllocInputTensor分配空间
  // 框架不进行输入与输出的Tensor分配
  // IN: input, param; OUT: input_tensors, output_tensors
  // 返回码：0，成功；否则失败；若存在申请失败，函数内部还需负责已申请空间的释放
  int PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors) override;

 private:
};
}  // namespace xstream
#endif  // MOBILENETV2PREDICTMETHOD_MOBILENETV2PREDICTMETHOD_H_
