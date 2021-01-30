/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: DnnPredictMethod.h
 * @Brief: declaration of the DnnPredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-11-23 11:09:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-11-23 16:01:33
 */

#ifndef INCLUDE_DNNPREDICTMETHOD_DNNPREDICTMETHOD_H_
#define INCLUDE_DNNPREDICTMETHOD_DNNPREDICTMETHOD_H_

#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include "json/json.h"
#include "horizon/vision_type/vision_type.hpp"
#include "hobotxstream/method.h"
#include "bpu_predict/bpu_predict_extension.h"
#include "DnnAsyncData.h"
namespace xstream {

class DnnPredictMethod : public Method {
 public:
  DnnPredictMethod() {}
  virtual ~DnnPredictMethod() {}

  int Init(const std::string &cfg_path) override;
  void Finalize() override;

  // 主逻辑，完全复用，派生类不需要再实现DoProcess
  std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<xstream::InputParamPtr> &param) override;

 public:
  Json::Value config_;
  std::string model_path_;

  std::shared_ptr<BPUModelWrapper> dnn_model_;
  bool dnn_is_sync_ = false;       // 默认异步,可配置
  bool dnn_run_with_roi_ = false;  // 默认非roi输入,可配置
  bool dnn_model_group_ = false;   // 是否开启group模式
  int dnn_model_group_id_ = 0;     // group模式，该模型的group id
  BPU_RUN_CTRL_S dnn_ctrl_;        // 运行的控制信息,core_id等
  int src_image_width_ = -1;
  int src_image_height_ = -1;

  int input_h_idx_, input_w_idx_, input_c_idx_;
  int model_input_height_, model_input_width_;

  // 申请InputTensor大小
  int AllocInputTensor(std::vector<BPU_TENSOR_S> &input_tensors);
  // 申请OutputTensor大小
  int AllocOutputTensor(std::vector<BPU_TENSOR_S> &output_tensors);
  int AllocOutputTensor(std::vector<BPU_TENSOR_S> &output_tensors, int num);

  // 释放InputTensor/OutputTensor
  void FreeTensor(std::vector<BPU_TENSOR_S> &tensors);

  // 派生类需要实现
  // PrepareInputData内部需要根据一帧图像目标数量，多次调用AllocInputTensor分配空间
  // 框架不进行输入与输出的Tensor分配
  // IN: input, param; OUT: input_tensors, output_tensors
  // 返回码：0，成功；否则失败；
  virtual int PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors) {
    return -1;
  }
  // 派生类需要实现
  // 将Method的输入预处理后，拷贝到金字塔以及roi
  // 该模式是特殊用法，只支持对所有的ROI打包一起，调用一次预测接口
  // X2和X3版本的金字塔数据结构不同，
  // 函数内部需要获取hobot::vision::PymImageFrame pyramid
  // IN: input, param; OUT: pyramid, input_bbox, valid_box, output_tensors
  // 返回码：0，成功；否则失败；若存在申请失败，函数内部还需负责已申请空间的释放
  virtual int PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      hobot::vision::PymImageFrame &pyramid,
      std::vector<BPU_BBOX> &input_bbox,
      std::vector<int> &valid_box,
      std::vector<BPU_TENSOR_S> &output_tensors) {
    return -1;
  }

  virtual int GetSrcImageSize(
      const std::vector<BaseDataPtr> &input,
      int &src_image_height,
      int &src_image_width) {
    src_image_height = src_image_height_;
    src_image_width = src_image_width_;
    return 0;
  }
};
}  // namespace xstream
#endif  // INCLUDE_DNNPREDICTMETHOD_DNNPREDICTMETHOD_H_
