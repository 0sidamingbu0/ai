/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: MultitaskPostProcessMethod.h
 * @Brief: declaration of the MultitaskPostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-03 19:32:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-03 21:21:32
 */

#ifndef INCLUDE_MULTITASKPOSTPROCESSMETHOD_MULTITASKPOSTPROCESSMETHOD_H_
#define INCLUDE_MULTITASKPOSTPROCESSMETHOD_MULTITASKPOSTPROCESSMETHOD_H_

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include "DnnAsyncData.h"
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "MultitaskPostProcessMethod/result_util.h"
#include "bpu_predict/bpu_parse_utils_extension.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class MultitaskPostProcessMethod : public DnnPostProcessMethod {
 public:
  MultitaskPostProcessMethod() {}
  virtual ~MultitaskPostProcessMethod() {}

  int Init(const std::string &cfg_path) override;

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  int ParseDnnResult(DnnAsyncData &dnn_result,
                     std::vector<BaseDataPtr> &frame_result) override;

 private:
  void ParseResults(DnnAsyncData &dnn_result, OutMsg &det_results);
  void GetRppRects(std::map<std::string, std::vector<BBox>> &boxes, int index,
                   const std::string &name,
                   const std::unordered_map<int, std::string> &labels,
                   BPU_MODEL_S *bpu_model,
                   std::vector<BPU_TENSOR_S> &output_tensors);
  void GetKps(BPU_MODEL_S bpu_model, int out_level,
              std::vector<Landmarks> &kpss, void *output,
              const std::vector<BBox> &body_boxes);
  void GetKps2(BPU_MODEL_S bpu_model,
               std::vector<hobot::vision::Landmarks> &kpss,
               void *kps_label_output, void *kps_offset_output,
               int kps_label_level, int kps_offset_level,
               const std::vector<hobot::vision::BBox> &body_boxes);
  void GetLMKS1(BPU_MODEL_S bpu_model, int out_level,
                std::vector<Landmarks> &landmarks, void *output,
                const std::vector<BBox> &face_boxes);
  void GetLMKS2(BPU_MODEL_S bpu_model, std::vector<Landmarks> &landmarks,
                void *lmks2_label_output, void *lmks2_offset_output,
                int lmks2_label_level, int lmks2_offset_level,
                const std::vector<BBox> &face_boxes);
  void GetPose(BPU_MODEL_S bpu_model, int out_level,
               std::vector<Pose3D> &face_pose, void *output,
               const std::vector<BBox> &face_boxes);
  void Convert2FrameworkData(
      OutMsg &det_results,
      std::map<std::string, std::shared_ptr<BaseDataVector>>
          &xstream_det_result);

 private:
  std::vector<std::string> method_outs_;
  std::map<int, BranchInfo> out_level2branch_info_;
  std::vector<ModelOutputInfo> model_output_info_;

  bool need_prepare_output_info_ = true;

  int model_input_width_;
  int model_input_height_;

  float kps_pos_distance_;
  int kps_feat_width_;
  int kps_feat_height_;
  int kps_points_number_;
  int kps_feat_stride_;
  float kps_anchor_param_;

  float lmk_pos_distance_;
  int lmk_feat_width_;
  int lmk_feat_height_;
  int lmk_feat_stride_;
  int lmk_points_number_;
  float lmk_anchor_param_;
  int face_pose_number_;
  float face_pv_thr_ = 0;

  float bbox_threshold_;

  int core_id_ = 2;
};

}  // namespace xstream
#endif  // INCLUDE_MULTITASKPOSTPROCESSMETHOD_MULTITASKPOSTPROCESSMETHOD_H_
