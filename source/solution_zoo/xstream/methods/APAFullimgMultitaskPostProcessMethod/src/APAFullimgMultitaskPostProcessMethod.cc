/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: APAFullimgMultitaskPostProcessMethod.cpp
 * @Brief: definition of the APAFullimgMultitaskPostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2021-01-08 15:29:05
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2021-01-08 20:43:08
 */

#include "APAFullimgMultitaskPostProcessMethod/APAFullimgMultitaskPostProcessMethod.h"
#include <string>
#include <vector>
#include "APAFullimgMultitaskPostProcessMethod/util.h"
#include "hobotlog/hobotlog.hpp"
#include "hobotxstream/profiler.h"
#include "dnn_util.h"
#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using hobot::vision::BBox;
using hobot::vision::Segmentation;

namespace xstream {

int APAFullimgMultitaskPostProcessMethod::Init(const std::string &cfg_path) {
  DnnPostProcessMethod::Init(cfg_path);
  person_box_score_thresh_ = config_.GetFloatValue("person_box_score_thresh",
                                                    person_box_score_thresh_);
  cycle_box_score_thresh_ = config_.GetFloatValue("cycle_box_score_thresh",
                                                   cycle_box_score_thresh_);
  rear_box_score_thresh_ = config_.GetFloatValue("rear_box_score_thresh",
                                                  rear_box_score_thresh_);
  vehicle_box_score_thresh_ =
      config_.GetFloatValue("vehicle_box_score_thresh",
                             vehicle_box_score_thresh_);
  parkinglock_box_score_thresh_ =
      config_.GetFloatValue("parkinglock_box_score_thresh",
                             parkinglock_box_score_thresh_);

  pre_nms_top_n_ = config_.GetIntValue("pre_nms_top_n");
  post_nms_top_n_ = config_.GetIntValue("post_nms_top_n");
  iou_threshold_ = config_.GetFloatValue("iou_threshold", iou_threshold_);
  return 0;
}

int APAFullimgMultitaskPostProcessMethod::ParseDnnResult(
    DnnAsyncData &dnn_result,
    std::vector<BaseDataPtr> &frame_result) {
  LOGD << "APAFullimgMultitaskPostProcessMethod ParseDnnResult";
  // 解析模型信息
  if (!modelinfo_is_init_) {
    bpu_model_ = dnn_result.dnn_model->bpu_model;
    src_img_width_ = dnn_result.src_image_width;
    src_img_height_ = dnn_result.src_image_height;
    GetOutputInfo(bpu_model_, layer2branch_info_);
    modelinfo_is_init_ = true;
  }

  // person_box, vehicle_box, rear_box, cycle_box, parkinglock_box, mask
  frame_result.resize(6, std::make_shared<xstream::BaseDataVector>());

  HOBOT_CHECK(dnn_result.output_tensors.size() == 1);  // 全图检测对应一次
  auto &output_tensors = dnn_result.output_tensors[0];
  if (static_cast<int>(output_tensors.size()) <
      bpu_model_.output_num) {
    LOGE << "OutputTensors size not match";
    return -1;
  }

  // flush cache
  for (int i = 0; i < bpu_model_.output_num; ++i) {
    HB_SYS_flushMemCache(&(output_tensors[i].data),
                         HB_SYS_MEM_CACHE_INVALIDATE);
  }

  // 模型输出13层，前12层解析5类检测框，第13层解析分割
  std::vector<std::vector<BBox>> boxes_result;
  ParseBox(bpu_model_, output_tensors, boxes_result);
  // 封装为BaseData
  for (int i = 0; i < 5; i++) {
    for (size_t j = 0; j < boxes_result[i].size(); ++j) {
      const auto &box = boxes_result[i][j];
      auto xstream_box = std::make_shared<XStreamData<BBox>>();
      xstream_box->value = std::move(box);
      std::static_pointer_cast<xstream::BaseDataVector>(
          frame_result[i])->datas_.push_back(xstream_box);
    }
  }
  Segmentation mask_result;
  ParseMask(bpu_model_, output_tensors, mask_result);
  // 封装为BaseData
  auto xstream_mask = std::make_shared<XStreamData<Segmentation>>();
  xstream_mask->value = std::move(mask_result);
  std::static_pointer_cast<xstream::BaseDataVector>(
      frame_result[5])->datas_.push_back(xstream_mask);
  return 0;
}

void APAFullimgMultitaskPostProcessMethod::GetOutputInfo(
    const BPU_MODEL_S &bpu_model,
    std::vector<BranchInfo> &branch_infos) {
  branch_infos.resize(bpu_model.output_num);
  for (size_t i = 0; i < branch_infos.size(); ++i) {
    // TODO(zhe.sun) BPU_LAYOUT_NCHW待处理
    HOBOT_CHECK(bpu_model.outputs[i].aligned_shape.layout == BPU_LAYOUT_NHWC);

    BranchInfo &branch_info = branch_infos[i];
    // shifts
    branch_info.shifts = bpu_model.outputs[i].shifts;
    // aligned_nhwc, real_nhwc
    int shape_dim = bpu_model.outputs[i].aligned_shape.ndim;
    HOBOT_CHECK(shape_dim == 4) << "shape_dim: " << shape_dim;
    branch_info.aligned_nhwc.resize(shape_dim);
    branch_info.real_nhwc.resize(shape_dim);
    for (int dim = 0; dim < shape_dim; dim++) {
      branch_info.aligned_nhwc[dim] = bpu_model.outputs[i].aligned_shape.d[dim];
      branch_info.real_nhwc[dim] = bpu_model.outputs[i].shape.d[dim];
    }
  }
}

void APAFullimgMultitaskPostProcessMethod::ParseBox(
    const BPU_MODEL_S &bpu_model,
    const std::vector<BPU_TENSOR_S> &output_tensors,
    std::vector<std::vector<BBox>> &boxes_result) {
  RUN_PROCESS_TIME_PROFILER("ParseFullImageBox");
  boxes_result.resize(5);  // 5类框
  // 第一层输出大小
  const int height_layer0 = layer2branch_info_[0].real_nhwc[1];
  const int width_layer0 = layer2branch_info_[0].real_nhwc[2];
  const std::vector<int> feat_stride = {8, 16, 32, 64};

  // 1. 计算fmap上每个点对应其在原图中的坐标locations : (h*w, 2)
  static std::vector<cv::Mat> locations(4);  // 分别对应stride=8,16,32,64
  static std::once_flag flag;
  std::call_once(flag, [&height_layer0, &width_layer0, &feat_stride]() {
    for (size_t i = 0; i < locations.size(); i++) {
      int scale = 1 << i;
      int height = height_layer0 / scale;
      int width = width_layer0 / scale;
      ComputeLocations(height, width, feat_stride[i], locations[i]);
    }
  });
  // box相关数据: *5 [person,vehicle,rear,cycle,parkinglock]
  std::vector<void*> bpu_out_scores = {
    output_tensors[0].data.virAddr,  // 88x160x5
    output_tensors[1].data.virAddr,  // 44x80x5
    output_tensors[2].data.virAddr,  // 22x40x4
    output_tensors[3].data.virAddr   // 11x20x4
  };
  std::vector<void*> bpu_out_regs = {
    output_tensors[4].data.virAddr,  // 88x160x20
    output_tensors[5].data.virAddr,  // 44x80x20
    output_tensors[6].data.virAddr,  // 22x40x16
    output_tensors[7].data.virAddr   // 11x20x16
  };
  std::vector<void*> bpu_out_ctrs = {
    output_tensors[8].data.virAddr,   // 88x160x5
    output_tensors[9].data.virAddr,   // 44x80x5
    output_tensors[10].data.virAddr,  // 22x40x4
    output_tensors[11].data.virAddr   // 11x20x4
  };

  // 2. 根据定点阈值，转换定点数据
  // scores_valid_index[i][j]: stride i, category j
  std::vector<std::vector<std::vector<int>>> scores_valid_index(4);
  std::vector<std::vector<std::vector<int>>> valid_index(4);
  std::vector<std::vector<std::vector<int32_t>>> scores_valid_float(4);
  std::vector<std::vector<std::vector<float>>> reg_valid_float(4);
  std::vector<std::vector<std::vector<int32_t>>> ctr_valid_float(4);

  for (int i = 0; i < 2; i++) {
    scores_valid_index[i].resize(5);
    valid_index[i].resize(5);
    scores_valid_float[i].resize(5);
    reg_valid_float[i].resize(5);
    ctr_valid_float[i].resize(5);  // person,vehicle,rear,cycle,parkinglock
  }
  for (int i = 2; i < 4; i++) {
    scores_valid_index[i].resize(4);
    valid_index[i].resize(4);
    scores_valid_float[i].resize(4);
    reg_valid_float[i].resize(4);
    ctr_valid_float[i].resize(4);  // person,vehicle,rear,cycle
  }

  // 2.1 先解析score, 过滤得到初步的valid_index
  float person_score_tmp_thresh =
      log(person_box_score_thresh_/(1.0-person_box_score_thresh_));
  float vehicle_score_tmp_thresh =
      log(vehicle_box_score_thresh_/(1.0-vehicle_box_score_thresh_));
  float rear_score_tmp_thresh =
      log(rear_box_score_thresh_/(1.0-rear_box_score_thresh_));
  float cycle_score_tmp_thresh =
      log(cycle_box_score_thresh_/(1.0-cycle_box_score_thresh_));
  float parkinglock_score_tmp_thresh =
      log(parkinglock_box_score_thresh_/(1.0-parkinglock_box_score_thresh_));
  std::vector<float> score_thresholds = {
      person_score_tmp_thresh,
      vehicle_score_tmp_thresh,
      rear_score_tmp_thresh,
      cycle_score_tmp_thresh,
      parkinglock_score_tmp_thresh};
  std::vector<float> score_thresholds_noparklock = {
      person_score_tmp_thresh,
      vehicle_score_tmp_thresh,
      rear_score_tmp_thresh,
      cycle_score_tmp_thresh};
  std::vector<int> channel_index = {
      0, 1, 2, 3, 4};
  std::vector<int> channel_index_noparklock = {
      0, 1, 2, 3};
  for (int i = 0; i < 2; i++) {
      ConvertOutputFilter(
          bpu_out_scores[i], i, score_thresholds, channel_index,
          scores_valid_index[i], scores_valid_float[i]);
  }
  for (int i = 2; i < 4; i++) {
    ConvertOutputFilter(
        bpu_out_scores[i], i, score_thresholds_noparklock,
        channel_index_noparklock, scores_valid_index[i],
        scores_valid_float[i]);
  }

  // 基于score的初步结果，解析ctr，并过滤得到valid_index
  for (int i = 0; i < 2; i++) {
    ConvertOutputFilterOfPrior(
        bpu_out_ctrs[i], i+8, score_thresholds, channel_index,
        scores_valid_index[i], valid_index[i],
        ctr_valid_float[i]);
  }
  for (int i = 2; i < 4; i++) {
    ConvertOutputFilterOfPrior(
        bpu_out_ctrs[i], i+8, score_thresholds_noparklock,
        channel_index_noparklock,
        scores_valid_index[i], valid_index[i],
        ctr_valid_float[i]);
  }
  // 更新scores_valid_float，使其与valid_index对齐
  for (size_t i = 0; i < valid_index.size(); i++) {  // stride i
  auto &valid_index_i = valid_index[i];
    for (size_t j = 0; j < valid_index_i.size(); j++) {  // category j
      auto valid_index_i_j = valid_index_i[j];
      size_t score_index = 0;
      for (size_t k = 0; k < valid_index_i_j.size(); k++) {
        while (scores_valid_index[i][j][score_index] != valid_index_i_j[k]) {
          score_index++;
          HOBOT_CHECK(score_index < scores_valid_index[i][j].size());
        }
        scores_valid_float[i][j][k] = scores_valid_float[i][j][score_index];
      }
      scores_valid_float[i][j].resize(valid_index_i_j.size());
    }
  }

  // 2.2 解析reg, 只需根据valid_index解析部分即可
  for (int i = 0; i < 4; i++) {  // stride i
    for (int j = 0; j < 5; j++) {  // category j
      if (j == 4 && i > 1) continue;

      ConvertOutputFilter(
          bpu_out_regs[i], i+4,
          j * 4, 4,
          valid_index[i][j],
          reg_valid_float[i][j]);
    }
  }

  // 2.3 分类计算score
  for (int j = 0; j < 5; j++) {  // category j
    std::vector<BBox> box_data;
    for (int i = 0; i < 4; i++) {  // stride i
      if (j == 4 && i > 1) continue;

      int valid_num = valid_index[i][j].size();
      if (valid_num == 0) continue;
      // score
      cv::Mat box_score_data_raw(valid_num, 1, CV_32SC1,
                                 scores_valid_float[i][j].data());
      // 转换浮点
      box_score_data_raw.convertTo(box_score_data_raw, CV_32FC1);
      box_score_data_raw /= (1 << layer2branch_info_[i].shifts[j]);
      cv::Mat box_ctr_data_raw(valid_num, 1, CV_32SC1,
                               ctr_valid_float[i][j].data());
      // 转换浮点
      box_ctr_data_raw.convertTo(box_ctr_data_raw, CV_32FC1);
      box_ctr_data_raw /= (1 << layer2branch_info_[i+8].shifts[j]);

      cv::Mat box_score_data, box_ctr_data;
      cv::exp(box_score_data_raw * (-1), box_score_data);
      box_score_data = 1 / (1 + box_score_data);
      cv::exp(box_ctr_data_raw * (-1), box_ctr_data);
      box_ctr_data = 1 / (1 + box_ctr_data);
      cv::multiply(box_score_data, box_ctr_data, box_score_data);
      // reg
      cv::Mat box_reg_data(valid_num, 4, CV_32FC1,
                           reg_valid_float[i][j].data());
      cv::exp(box_reg_data, box_reg_data);
      box_reg_data = box_reg_data * (1 << (i+3));

      // 根据score阈值过滤
      for (int row = 0; row < box_score_data.rows; row++) {
        float* ptr = box_score_data.ptr<float>(row);
        for (int col = 0; col < box_score_data.cols; col++) {
          if (ptr[col] > person_box_score_thresh_) {
            BBox one_box_data;
            one_box_data.score = ptr[col];
            int location_0, location_1;
            location_0 = locations[i].ptr<int32_t>(
                valid_index[i][j][row])[0];
            location_1 = locations[i].ptr<int32_t>(
                valid_index[i][j][row])[1];
            // 限制框大小在图像内
            one_box_data.x1 = std::max(
                std::min(location_0 - box_reg_data.ptr<float>(row)[0],
                         static_cast<float>(src_img_width_) - 1), 0.0f);
            one_box_data.y1 = std::max(
                std::min(location_1 - box_reg_data.ptr<float>(row)[1],
                         static_cast<float>(src_img_height_) - 1), 0.0f);
            one_box_data.x2 = std::max(
                std::min(location_0 + box_reg_data.ptr<float>(row)[2],
                         static_cast<float>(src_img_width_) - 1), 0.0f);
            one_box_data.y2 = std::max(
                std::min(location_1 + box_reg_data.ptr<float>(row)[3],
                         static_cast<float>(src_img_height_) - 1), 0.0f);
            box_data.push_back(one_box_data);
          }
        }
      }
    }  // end stride i
    // NMS box
    int pre_nms_top_n = box_data.size();
    if (pre_nms_top_n > pre_nms_top_n_) {  // 需要取前top_n_个
      auto greater = [](const BBox &a, const BBox &b) {
        return a.score > b.score;
      };
      std::sort(box_data.begin(), box_data.end(), greater);  // 降序排序
      box_data.resize(pre_nms_top_n_);
    }
    std::vector<BBox> box_result;
    LocalIOU(box_data, box_result, iou_threshold_, post_nms_top_n_);
    LOGD << j << " category box size: " << box_result.size();
    for (size_t i = 0; i < box_result.size(); i++) {
      LOGD << box_result[i];
    }
    boxes_result[j] = box_result;
  }  // end category j
  return;
}

void APAFullimgMultitaskPostProcessMethod::ConvertOutputFilter(
    void *src_ptr,
    int out_index,
    std::vector<float> threshold,
    std::vector<int> channel,
    std::vector<std::vector<int>> &valid_index,
    std::vector<std::vector<int32_t>> &out_data) {
  HOBOT_CHECK(channel.size() == threshold.size());
  auto &aligned_shape = bpu_model_.outputs[out_index].aligned_shape;
  auto &real_shape = bpu_model_.outputs[out_index].shape;
  HOBOT_CHECK(static_cast<int>(channel.size()) <= real_shape.d[3]);
  HOBOT_CHECK(bpu_model_.outputs[out_index].data_type == BPU_TYPE_TENSOR_S32);

  auto shift = bpu_model_.outputs[out_index].shifts;

  for (size_t i = 0; i < channel.size(); i++) {
    threshold[i] *= (1 << shift[channel[i]]);  // 转换浮点阈值到定点阈值
  }

  // 初始化
  int index = 0;
  valid_index.resize(channel.size());
  out_data.resize(channel.size());

  uint32_t src_n_stride =
      aligned_shape.d[1] * aligned_shape.d[2] * aligned_shape.d[3];
  uint32_t src_h_stride = aligned_shape.d[2] * aligned_shape.d[3];
  uint32_t src_w_stride = aligned_shape.d[3];

  int32_t tmp_int32_value;

  for (int nn = 0; nn < real_shape.d[0]; nn++) {
    void *cur_n_src = reinterpret_cast<int32_t *>(src_ptr) + nn * src_n_stride;
    for (int hh = 0; hh < real_shape.d[1]; hh++) {
      void *cur_h_src =
          reinterpret_cast<int32_t *>(cur_n_src) + hh * src_h_stride;
      for (int ww = 0; ww < real_shape.d[2]; ww++) {
        void *cur_w_src =
            reinterpret_cast<int32_t *>(cur_h_src) + ww * src_w_stride;
        for (size_t cc = 0; cc < channel.size(); cc++) {
          tmp_int32_value = *(reinterpret_cast<int32_t *>(
              cur_w_src) + channel[cc]);
          if (tmp_int32_value > threshold[cc]) {
            out_data[cc].push_back(tmp_int32_value);
            valid_index[cc].push_back(index);
          }
        }
        index++;
      }
    }
  }
}

void APAFullimgMultitaskPostProcessMethod::ConvertOutputFilterOfPrior(
    void *src_ptr,
    int out_index,
    std::vector<float> threshold,
    std::vector<int> channel,
    const std::vector<std::vector<int>> &previous_valid_index,
    std::vector<std::vector<int>> &valid_index,
    std::vector<std::vector<int32_t>> &out_data) {
  HOBOT_CHECK(channel.size() == threshold.size() &&
              channel.size() == previous_valid_index.size());
  auto &aligned_shape = layer2branch_info_[out_index].aligned_nhwc;
  auto &real_shape = layer2branch_info_[out_index].real_nhwc;
  HOBOT_CHECK(static_cast<int>(channel.size()) <= real_shape[3]);
  HOBOT_CHECK(bpu_model_.outputs[out_index].data_type == BPU_TYPE_TENSOR_S32);

  auto shift = bpu_model_.outputs[out_index].shifts;

  for (size_t i = 0; i < channel.size(); i++) {
    threshold[i] *= (1 << shift[channel[i]]);  // 转换浮点阈值到定点阈值
  }

  int src_h_stride = aligned_shape[2] * aligned_shape[3];
  int src_w_stride = aligned_shape[3];

  // 初始化
  valid_index.resize(channel.size());
  out_data.resize(channel.size());

  for (size_t cc = 0; cc < previous_valid_index.size(); cc++) {
    auto &previous_valid_index_channel = previous_valid_index[cc];
    for (size_t i = 0; i < previous_valid_index_channel.size(); i++) {
      int index = previous_valid_index_channel[i];
      // 根据index计算h, w
      int h_index = index / real_shape[2];  // 有效size
      int w_index = index % real_shape[2];
      // 对有效索引处的数据定点转浮点
      int32_t tmp_int32_value = *(reinterpret_cast<int32_t *>(
          src_ptr) + h_index * src_h_stride +
          w_index * src_w_stride + channel[cc]);
      if (tmp_int32_value > threshold[cc]) {
        out_data[cc].push_back(tmp_int32_value);
        valid_index[cc].push_back(index);
      }
    }
  }
}

void APAFullimgMultitaskPostProcessMethod::ConvertOutputFilter(
    void *src_ptr,
    int out_index,
    const int channel, const int step,
    const std::vector<int> &valid_index,
    std::vector<float> &out_data) {
  auto &aligned_shape = layer2branch_info_[out_index].aligned_nhwc;
  auto &real_shape = layer2branch_info_[out_index].real_nhwc;
  HOBOT_CHECK(bpu_model_.outputs[out_index].data_type == BPU_TYPE_TENSOR_S32);

  auto shift = bpu_model_.outputs[out_index].shifts;

  uint32_t src_h_stride = aligned_shape[2] * aligned_shape[3];
  uint32_t src_w_stride = aligned_shape[3];

  out_data.resize(valid_index.size() * step);

  for (size_t i = 0; i < valid_index.size(); i++) {
    int index = valid_index[i];
    // 根据index计算h,w
    int h_index = index / real_shape[2];  // 有效宽度
    int w_index = index % real_shape[2];
    // 对有效索引处的数据定点转浮点
    for (int ss = 0; ss < step; ss++) {
      int32_t tmp_int32_value = *(reinterpret_cast<int32_t *>(
          src_ptr) + h_index * src_h_stride +
          w_index * src_w_stride + channel + ss);
      float tmp_float_value = GetFloatByInt(tmp_int32_value, shift[channel+ss]);
      out_data[i*step + ss] = tmp_float_value;
    }
  }
}

void APAFullimgMultitaskPostProcessMethod::LocalIOU(
    std::vector<BBox> &candidates,
    std::vector<BBox> &result,
    const float overlap_ratio, const int top_N) {
  if (candidates.size() == 0) {
    return;
  }
  std::vector<bool> skip(candidates.size(), false);

  auto greater = [](const BBox &a, const BBox &b) {
    return a.score > b.score;
  };
  std::stable_sort(candidates.begin(), candidates.end(),
                   greater);

  int count = 0;
  for (size_t i = 0; count < top_N && i < skip.size(); ++i) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    float area_i = candidates[i].Width() * candidates[i].Height();

    // suppress the significantly covered bbox
    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      // get intersections
      float xx1 =
          std::max(candidates[i].x1, candidates[j].x1);
      float yy1 =
          std::max(candidates[i].y1, candidates[j].y1);
      float xx2 =
          std::min(candidates[i].x2, candidates[j].x2);
      float yy2 =
          std::min(candidates[i].y2, candidates[j].y2);
      float area_intersection = (xx2 - xx1) * (yy2 - yy1);
      bool area_intersection_valid = (area_intersection > 0) && (xx2 - xx1 > 0);

      if (area_intersection_valid) {
        // compute overlap
        float area_j = candidates[j].Width() * candidates[j].Height();
        float o = area_intersection / (area_i + area_j - area_intersection);

        if (o > overlap_ratio) {
          skip[j] = true;
        }
      }
    }
    result.push_back(candidates[i]);
  }
}

void APAFullimgMultitaskPostProcessMethod::ComputeLocations(
    const int &height,
    const int &width,
    const int &feature_stride,
    cv::Mat &location) {
  std::vector<int32_t> shift_x = Arange(0, width);
  std::vector<int32_t> shift_y = Arange(0, height);
  cv::Mat shift_x_mat = cv::Mat(shift_x) * feature_stride;
  cv::Mat shift_y_mat = cv::Mat(shift_y) * feature_stride;
  // meshgrid
  cv::Mat shift_x_mesh, shift_y_mesh;
  MeshGrid(shift_x_mat, shift_y_mat, shift_x_mesh, shift_y_mesh);
  cv::Mat concat_xy;
  cv::vconcat(shift_x_mesh.reshape(1, 1),
              shift_y_mesh.reshape(1, 1),
              concat_xy);    // 纵向拼接两个行向量
  cv::transpose(concat_xy, location);
  location = location + feature_stride / 2;  // 向下取整
}

void APAFullimgMultitaskPostProcessMethod::ParseMask(
    const BPU_MODEL_S &bpu_model,
    const std::vector<BPU_TENSOR_S> &output_tensors,
    Segmentation &mask_result) {
  RUN_PROCESS_TIME_PROFILER("ParseFullImageMask");
  int out_branch = 12;
  int8_t* mask_feature =
      reinterpret_cast<int8_t *>(output_tensors[out_branch].data.virAddr);
  int sparse_height = layer2branch_info_[12].real_nhwc[1];
  int sparse_width = layer2branch_info_[12].real_nhwc[2];  // 需要最近邻插值
  LOGD << "sparse_height: " << sparse_height;
  LOGD << "sparse_width: " << sparse_width;
  mask_result.height = sparse_height;
  mask_result.width = sparse_width;

  cv::Mat sparse_mask(sparse_height, sparse_width, CV_8SC1, mask_feature);
  // debug
  // cv::imwrite("sparse_mask.png", sparse_mask);
  cv::Mat mask_float;
  sparse_mask.convertTo(mask_float, CV_32FC1);
  mask_result.values = std::vector<float>(mask_float.reshape(1, 1));
}

}  // namespace xstream
