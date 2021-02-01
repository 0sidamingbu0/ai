/*
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: PlateNumPostProcessMethod.cpp
 * @Brief: implementation of PlateNumPostProcessMethod
 * @Author: shiyu.fu
 * @Email: shiyu.fu@horizon.ai
 * @Date: Mon Jan 04 2021 13:56:36
 */

#include "PlateNumPostProcessMethod/PlateNumPostProcessMethod.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "DnnAsyncData.h"
#include "dnn_util.h"
#include "hobotlog/hobotlog.hpp"
#include "hobotxstream/profiler.h"
#include "horizon/vision_type/vision_type.hpp"

namespace xstream {

int PlateNumPostProcessMethod::Init(const std::string &cfg_path) {
  LOGD << "PlateNumPostProcessMethod Init";
  DnnPostProcessMethod::Init(cfg_path);

  output_size_ = config_.GetIntValue("output_size", 0);
  HOBOT_CHECK(output_size_ > 0) << "error output size: " << output_size_;

  return 0;
}

int PlateNumPostProcessMethod::ParseDnnResult(
    DnnAsyncData &dnn_result, std::vector<BaseDataPtr> &frame_result) {
  LOGD << "PlateNumPostProcessMethod ParseDnnResult";
  // currently only support single row

  if (dnn_result.output_tensors.empty()) {
    LOGD << "failed to run model in predict method";
    frame_result.resize(output_size_);

    for (uint32_t i = 0; i < output_size_; ++i) {
      auto base_data_vector = std::make_shared<BaseDataVector>();
      frame_result[i] = std::static_pointer_cast<BaseData>(base_data_vector);
    }

    return 0;
  }

  {
    RUN_PROCESS_TIME_PROFILER("PlateNumPostProcess");
    RUN_FPS_PROFILER("PlateNumPostProcess");
    frame_result.resize(output_size_);
    auto &output_tensors = dnn_result.output_tensors[0];
    int output_num = dnn_result.dnn_model->bpu_model.output_num;
    LOGD << "output num: " << output_num;

    // convert to float
    int feature_size = 1;
    for (int i = 0;
         i < dnn_result.dnn_model->bpu_model.outputs[0].aligned_shape.ndim;
         ++i) {
      feature_size *=
          dnn_result.dnn_model->bpu_model.outputs[0].aligned_shape.d[i];
    }
    if (dnn_result.dnn_model->bpu_model.outputs[0].data_type ==
            BPU_TYPE_TENSOR_F32 ||
        dnn_result.dnn_model->bpu_model.outputs[0].data_type ==
            BPU_TYPE_TENSOR_S32 ||
        dnn_result.dnn_model->bpu_model.outputs[0].data_type ==
            BPU_TYPE_TENSOR_U32) {
      feature_size *= 4;
    }
    std::vector<std::vector<float>> features;
    for (size_t i = 0, feat_idx = 0; i < dnn_result.valid_box.size(); ++i) {
      if (dnn_result.valid_box[i] == 1) {
        std::vector<float> feature(feature_size);
        auto src_ptr =
            reinterpret_cast<int8_t *>(output_tensors[0].data.virAddr) +
            feat_idx * feature_size;
        int ret = ConvertOutputToFloat(src_ptr, feature.data(),
                                       dnn_result.dnn_model->bpu_model, 0);
        if (ret == -1) {
          LOGI << "output data type is float, no need to convert";
        }
        features.push_back(feature);
        feat_idx++;
      }
    }

    for (uint32_t out_idx = 0; out_idx < output_size_; ++out_idx) {
      auto base_data_vector = std::make_shared<BaseDataVector>();
      ParsePlateNum(dnn_result.dnn_model->bpu_model, dnn_result.valid_box,
                    features, base_data_vector);
      frame_result[out_idx] =
          std::static_pointer_cast<BaseData>(base_data_vector);
    }
  }

  return 0;
}

void PlateNumPostProcessMethod::ParsePlateNum(
    BPU_MODEL_S bpu_model, std::vector<int> &valid_box,
    std::vector<std::vector<float>> &features,
    std::shared_ptr<BaseDataVector> result) {
  int real_shape_dim = bpu_model.outputs[0].shape.ndim;
  HOBOT_CHECK(real_shape_dim == 4) << "error aligned shape dim: "
                                   << real_shape_dim;
  auto real_shape = bpu_model.outputs[0].shape.d;
  int slice_num = real_shape[2];
  int score_num = real_shape[3];

  for (size_t roi_idx = 0, feature_idx = 0; roi_idx < valid_box.size();
       ++roi_idx) {
    auto plate_num = std::make_shared<XStreamData<std::vector<int>>>();
    if (valid_box[roi_idx] != 1) {
      plate_num->value.clear();
    } else {
      std::vector<int> nums;

      // get max index for each slice
      for (int slice_idx = 0; slice_idx < slice_num; ++slice_idx) {
        int max_index = -1;
        float max_score = -999.0f;

        for (int score_idx = 0; score_idx < score_num; ++score_idx) {
          int idx = slice_idx * score_num + score_idx;
          float cur_score = features[feature_idx][idx];
          if (cur_score > max_score) {
            max_index = score_idx;
            max_score = cur_score;
          }
        }

        nums.push_back(max_index);
      }

      // skip splitted/repeated characters
      int last_index = -1;
      for (auto iter = nums.begin(); iter != nums.end();) {
        if (*iter == last_index || *iter == 0) {
          last_index = *iter;
          iter = nums.erase(iter);
        } else {
          last_index = *iter;
          iter++;
        }
      }

      plate_num->value.assign(nums.begin(), nums.end());

      // plate num can only be 7 or 8 digits
      if (plate_num->value.size() != 7 && plate_num->value.size() != 8) {
        plate_num->value.clear();
      }

#ifdef DEBUG
      LOGD << "plate num: ";
      for (auto item : plate_num->value) {
        LOGD << item;
      }
#endif

      feature_idx++;
    }
    result->datas_.push_back(plate_num);
  }
  LOGD << "result->datas_.size(): " << result->datas_.size();
}

}  // namespace xstream
