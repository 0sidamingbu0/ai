/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: yolov3_predict_method.h
 * @Brief: declaration of the Yolov3PredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 11:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 18:21:33
 */

#ifndef YOLOV3PREDICTMETHOD_YOLOV3PREDICTMETHOD_H_
#define YOLOV3PREDICTMETHOD_YOLOV3PREDICTMETHOD_H_

#include <string>
#include <vector>
#include "DnnPredictMethod/DnnPredictMethod.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class Yolov3PredictMethod : public DnnPredictMethod {
 public:
  Yolov3PredictMethod() {}
  virtual ~Yolov3PredictMethod() {}

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

  int GetSrcImageSize(
      const std::vector<BaseDataPtr> &input,
      int &src_image_height,
      int &src_image_width) override;

 private:
  int pyramid_layer_ = 0;  // 默认使用金字塔第0层
};
}  // namespace xstream
#endif  // YOLOV3PREDICTMETHOD_YOLOV3PREDICTMETHOD_H_
