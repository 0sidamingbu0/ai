/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: yolov3_post_process_method.h
 * @Brief: declaration of the Yolov3PostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 17:07:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 19:21:33
 */

#ifndef YOLOV3POSTPROCESSMETHOD_YOLOV3POSTPROCESSMETHOD_H_
#define YOLOV3POSTPROCESSMETHOD_YOLOV3POSTPROCESSMETHOD_H_

#include <string>
#include <vector>
#include <utility>
#include "DnnPostProcessMethod/DnnPostProcessMethod.h"
#include "horizon/vision_type/vision_type.hpp"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

struct Detection;

class Yolov3PostProcessMethod : public DnnPostProcessMethod {
 public:
  Yolov3PostProcessMethod() {}
  virtual ~Yolov3PostProcessMethod() {}

  int Init(const std::string &cfg_path) override;

  // 派生类需要实现
  // 完成模型的后处理，以及转换成Method输出格式
  // IN: dnn_result. OUT: frame_result
  int ParseDnnResult(DnnAsyncData &dnn_result,
                     std::vector<BaseDataPtr> &frame_result) override;

  void nms(std::vector<Detection> &input, float iou_threshold,
           int top_k, std::vector<Detection> &result, bool suppress);

 private:
  float score_threshold_ = 0.3;
  float nms_threshold_ = 0.45;
  int nms_top_k_ = 500;

  // 用于坐标映射
  // 需要配置，且与预测阶段选取金字塔基础层大小一致
  int basic_pyramid_image_height_;
  int basic_pyramid_image_width_;
};

// Yolo3Config
struct Yolo3Config {
  std::vector<int> strides;
  std::vector<std::vector<std::pair<double, double>>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;

  std::string stridesToString() {
    std::stringstream ss;
    for (const auto &stride : strides) {
      ss << stride << " ";
    }
    return ss.str();
  }

  std::string anchorsTableToString() {
    std::stringstream ss;
    for (const auto &anchors : anchors_table) {
      for (auto data : anchors) {
        ss << "[" << data.first << "," << data.second << "] ";
      }
    }
    return ss.str();
  }
};

struct Detection {
  hobot::vision::BBox bbox;
  Detection() {}
  ~Detection() {}

  explicit Detection(hobot::vision::BBox bbox) : bbox(bbox) {}

  friend bool operator>(const Detection &lhs, const Detection &rhs) {
    return (lhs.bbox.score > rhs.bbox.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection &det) {
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("class_name")"
       << ":" << det.bbox.category_name << ","
       << R"("id")"
       << ":" << det.bbox.id << "}";
    return os;
  }
};

}  // namespace xstream
#endif  // YOLOV3POSTPROCESSMETHOD_YOLOV3POSTPROCESSMETHOD_H_
