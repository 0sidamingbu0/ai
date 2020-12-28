/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: mobilenetv2_post_process_method.h
 * @Brief: declaration of the Mobilenetv2PostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 17:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 19:21:33
 */

#ifndef MOBILENETV2POSTPROCESSMETHOD_MOBILENETV2POSTPROCESSMETHOD_H_
#define MOBILENETV2POSTPROCESSMETHOD_MOBILENETV2POSTPROCESSMETHOD_H_

#include <string>
#include <vector>
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "horizon/vision_type/vision_type.hpp"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class Mobilenetv2PostProcessMethod : public DnnPostProcessMethod {
 public:
  Mobilenetv2PostProcessMethod() {}
  virtual ~Mobilenetv2PostProcessMethod() {}

  int Init(const std::string &cfg_path) override;

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  int ParseDnnResult(DnnAsyncData &dnn_result,
                     std::vector<BaseDataPtr> &frame_result) override;

  void GetMaxResult(
    BPU_TENSOR_S &tensor, hobot::vision::BBox &cls);

 private:
  std::vector<std::string> class_names_;  // 分类标签，需要从文件读取
};

}  // namespace xstream
#endif  // YOLOV3POSTPROCESSMETHOD_YOLOV3POSTPROCESSMETHOD_H_
