/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: MattingTrimapFreePostProcessMethod.h
 * @Brief: declaration of the MattingTrimapFreePostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-15 19:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-15 20:21:33
 */

#ifndef INCLUDE_MATTINGTRIMAPFREEPOSTPROCESSMETHOD_MATTINGTRIMAPFREEPOSTPROCESSMETHOD_H_  // NOLINT
#define INCLUDE_MATTINGTRIMAPFREEPOSTPROCESSMETHOD_MATTINGTRIMAPFREEPOSTPROCESSMETHOD_H_  // NOLINT

#include <string>
#include <vector>
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "bpu_predict/bpu_predict_extension.h"
#ifdef X3
#include "./bpu_predict_x3.h"
#endif
namespace xstream {

class MattingTrimapFreePostProcessMethod : public DnnPostProcessMethod {
 public:
  MattingTrimapFreePostProcessMethod() {}
  virtual ~MattingTrimapFreePostProcessMethod() {}

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  virtual int ParseDnnResult(DnnAsyncData &dnn_result,
                             std::vector<BaseDataPtr> &frame_result);
};
}  // namespace xstream
#endif
// INCLUDE_MATTINGTRIMAPFREEPOSTPROCESSMETHOD_MATTINGTRIMAPFREEPOSTPROCESSMETHOD_H_
