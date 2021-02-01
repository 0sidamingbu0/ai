/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: mobilenetv2_post_process_method.cc
 * @Brief: definition of the Mobilenetv2PostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 11:12:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 21:21:33
 */

#include "Mobilenetv2PostProcessMethod/mobilenetv2_post_process_method.h"
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "opencv2/highgui/highgui.hpp"
#include "hobotxstream/profiler.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "horizon/vision_type/vision_type.hpp"
#include "hobotxstream/image_tools.h"
#include "hobotlog/hobotlog.hpp"
#include "DnnAsyncData.h"
#include "common/com_func.h"

namespace xstream {

int Mobilenetv2PostProcessMethod::Init(const std::string &cfg_path) {
  DnnPostProcessMethod::Init(cfg_path);
  std::string cls_names_list =
      config_.GetSTDStringValue("class_names_list_path");

  cls_names_list = hobotcommon::get_parent_path(cfg_path) + cls_names_list;
  std::ifstream fi(cls_names_list);
  if (fi) {
    std::string line;
    while (std::getline(fi, line)) {
      class_names_.push_back(line);
    }
  } else {
    LOGE << "load class_names_list faild, file_path: "
         << cls_names_list;
    return -1;
  }
  return 0;
}

int Mobilenetv2PostProcessMethod::ParseDnnResult(
    DnnAsyncData &dnn_result,
    std::vector<BaseDataPtr> &frame_result) {
  LOGD << "Mobilenetv2PostProcessMethod ParseDnnResult";
  auto xstream_det_result = std::make_shared<xstream::BaseDataVector>();

  // 一个检测框对应一次预测
  std::vector<hobot::vision::BBox> result;
  for (size_t i = 0; i < dnn_result.output_tensors.size(); i++) {
    auto &tensors = dnn_result.output_tensors[i];

    if (tensors.size() == 0) {  // 预处理或预测失败
      auto xstream_box = std::make_shared<
          xstream::XStreamData<hobot::vision::BBox>>();
      xstream_box->state_ = DataState::INVALID;
      xstream_det_result->datas_.push_back(xstream_box);
      continue;
    }

    HOBOT_CHECK(tensors.size() == 1);  // 模型仅一层输出

    hobot::vision::BBox cls;
    GetMaxResult(tensors[0], cls);
    LOGD << "id: " << cls.id
         << ", category_name: " << cls.category_name;

    // 转换BaseData
    auto xstream_box = std::make_shared<
        xstream::XStreamData<hobot::vision::BBox>>();
    xstream_box->value = std::move(cls);
    xstream_det_result->datas_.push_back(xstream_box);
  }
  frame_result.push_back(xstream_det_result);
  return 0;
}

void Mobilenetv2PostProcessMethod::GetMaxResult(
    BPU_TENSOR_S &tensor, hobot::vision::BBox &cls) {
  HB_SYS_flushMemCache(&(tensor.data), HB_SYS_MEM_CACHE_INVALIDATE);
  float *scores = reinterpret_cast<float *>(tensor.data.virAddr);
  float score = 0;
  int id = 0;
  int *shape = tensor.data_shape.d;
  for (auto i = 0; i < shape[1] * shape[2] * shape[3]; i++) {
    if (scores[i] > score) {
      score = scores[i];
      id = i;
    }
  }
  cls.id = id;
  cls.score = score;
  if (!class_names_.empty()) {
    cls.category_name = class_names_[id];
  }
}

}  // namespace xstream
