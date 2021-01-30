/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: APAFullimgMultitaskPredictMethod.h
 * @Brief: declaration of the APAFullimgMultitaskPredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2021-01-08 14:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2021-01-08 15:37:18
 */

#ifndef INCLUDE_APAFULLIMGMULTITASKPREDICTMETHOD_APAFULLIMGMULTITASKPREDICTMETHOD_H_  // NOLINT
#define INCLUDE_APAFULLIMGMULTITASKPREDICTMETHOD_APAFULLIMGMULTITASKPREDICTMETHOD_H_  // NOLINT

#include <string>
#include <vector>
#include "DnnPredictMethod/DnnPredictMethod.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class APAFullimgMultitaskPredictMethod : public DnnPredictMethod {
 public:
  APAFullimgMultitaskPredictMethod() {}
  virtual ~APAFullimgMultitaskPredictMethod() {}

  int Init(const std::string &cfg_path) override;

  int GetSrcImageSize(
    const std::vector<BaseDataPtr> &input,
    int &src_image_height,
    int &src_image_width) override;

  // 派生类需要实现
  // PrepareInputData内部需要根据一帧图像目标数量，多次调用AllocInputTensor分配空间
  // 框架不进行输入与输出的Tensor分配
  // IN: input, param; OUT: input_tensors, output_tensors
  // 返回码：0，成功；否则失败；若存在申请失败，函数内部还需负责已申请空间的释放
  virtual int PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors);

 private:
  int pyramid_layer_ = 0;
};
}  // namespace xstream
#endif
// INCLUDE_APAFULLIMGMULTITASKPREDICTMETHOD_APAFULLIMGMULTITASKPREDICTMETHOD_H_
