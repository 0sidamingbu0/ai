/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: APAFullimgMultitaskPostProcessMethod.h
 * @Brief: declaration of the APAFullimgMultitaskPostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2021-01-08 15:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2021-01-08 15:29:33
 */

#ifndef INCLUDE_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_H_  // NOLINT
#define INCLUDE_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_H_  // NOLINT

#include <string>
#include <vector>
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "horizon/vision_type/vision_type.hpp"
#include "bpu_predict/bpu_predict_extension.h"

using hobot::vision::BBox;
using hobot::vision::Segmentation;
namespace xstream {

struct BranchInfo {
  std::vector<int> real_nhwc;
  std::vector<int> aligned_nhwc;
  uint8_t *shifts;
};

class APAFullimgMultitaskPostProcessMethod : public DnnPostProcessMethod {
 public:
  APAFullimgMultitaskPostProcessMethod() {}
  virtual ~APAFullimgMultitaskPostProcessMethod() {}

  int Init(const std::string &cfg_path) override;
  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  virtual int ParseDnnResult(DnnAsyncData &dnn_result,
                             std::vector<BaseDataPtr> &frame_result);

 private:
  float person_box_score_thresh_ = 0.1629;
  float cycle_box_score_thresh_ = 0.1825;
  float rear_box_score_thresh_ = 0.1423;
  float vehicle_box_score_thresh_ = 0.1255;
  float parkinglock_box_score_thresh_ = 0.1740;

  int pre_nms_top_n_ = 4095;
  int post_nms_top_n_ = 100;
  float iou_threshold_ = 0.5;

  BPU_MODEL_S bpu_model_;
  int src_img_height_, src_img_width_;         // 原图大小
  std::vector<BranchInfo> layer2branch_info_;  // 模型输出信息
  bool modelinfo_is_init_ = false;             // 输出信息初始化标志

  // 获取模型输出信息
  void GetOutputInfo(const BPU_MODEL_S &bpu_model,
                     std::vector<BranchInfo> &branch_infos);

  // 解析检测框
  void ParseBox(
    const BPU_MODEL_S &bpu_model,
    const std::vector<BPU_TENSOR_S> &output_tensors,
    std::vector<std::vector<BBox>> &boxes_result);

  // 计算fmap上每个点到原图的坐标
  // IN: height, width, feature_stride:输出大小以及到原图的scale
  // OUT: location对应到原图的坐标, [height*width, 2]
  static void ComputeLocations(
    const int &height,
    const int &width,
    const int &feature_stride,
    cv::Mat &location);

  // 输出有效定点数据，后续需要转浮点
  // IN: src_ptr【定点数据地址】, out_index【输出数据所在层数】
  // IN: channel【需要转换的通道索引】,与threshold需要一一对应
  // IN: threshold【浮点阈值】:
  // 只有定点数据大于等于threshold*(2^^shift)，才认为有效
  // OUT: valid_index【满足阈值的定点的在channel上的有效索引(按hw排序)】
  // OUT: out_data【有效定点数据】
  // threshold.size() = channel.size() = valid_index.size() = out_data.size()
  void ConvertOutputFilter(void *src_ptr,
                           int out_index,
                           std::vector<float> threshold,
                           std::vector<int> channel,
                           std::vector<std::vector<int>> &valid_index,
                           std::vector<std::vector<int32_t>> &out_data);

  // 输出有效定点数据，后续需要转浮点
  // IN: src_ptr【定点数据地址】, out_index【输出数据所在层数】
  // IN: channel【需要转换的通道索引】,与threshold需要一一对应
  // IN: threshold【浮点阈值】:
  // 只有定点数据大于等于threshold*(2^^shift)，才认为有效
  // IN: previous_valid_index【需要转换的有效索引(按hw排序)】
  // OUT: valid_index【满足阈值的定点的在channel上的有效索引(按hw排序)】
  // OUT: out_data【有效定点数据】
  // threshold.size() = channel.size() = valid_index.size()...
  // = out_data.size() = previous_valid_index.size()
  void ConvertOutputFilterOfPrior(
      void *src_ptr, int out_index,
      std::vector<float> threshold,
      std::vector<int> channel,
      const std::vector<std::vector<int>> &previous_valid_index,
      std::vector<std::vector<int>> &valid_index,
      std::vector<std::vector<int32_t>> &out_data);

  // 定点转浮点
  // IN: src_ptr【定点数据地址】, out_index【输出数据所在层数】
  // IN: channel【需要转换的通道索引】
  // IN: step【表示从转换[channel:channel+step)通道数据】
  // IN: valid_index【满足阈值的定点的在channel上的有效索引(按hw排序)】
  // OUT: out_data【浮点数据】
  void ConvertOutputFilter(
    void *src_ptr,
    int out_index,
    const int channel, const int step,
    const std::vector<int> &valid_index,
    std::vector<float> &out_data);

  // 对box进行NMS
  // IN: candidates【输入的BBox】
  // IN: overlap_ratio【交并比】, top_N【输出最大数量】
  // OUT: result【输出结果】
  void LocalIOU(std::vector<BBox> &candidates,
                std::vector<BBox> &result,
                const float overlap_ratio, const int top_N);

  // 解析全图分割结果
  void ParseMask(
      const BPU_MODEL_S &bpu_model,
      const std::vector<BPU_TENSOR_S> &output_tensors,
      Segmentation &mask_result);
};
}  // namespace xstream
#endif
// INCLUDE_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_H_
