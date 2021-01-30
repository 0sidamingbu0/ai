/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: yolov3_post_process_method.cc
 * @Brief: definition of the Yolov3PostProcessMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 11:12:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 21:21:33
 */

#include "Yolov3PostProcessMethod/yolov3_post_process_method.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "hobotxstream/profiler.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "horizon/vision_type/vision_type.hpp"
#include "hobotxstream/image_tools.h"
#include "hobotlog/hobotlog.hpp"
#include "DnnAsyncData.h"

namespace xstream {

// model config, label etc.
Yolo3Config default_yolo3_config = {
    {32, 16, 8},
    {{{3.625, 2.8125}, {4.875, 6.1875}, {11.65625, 10.1875}},
     {{1.875, 3.8125}, {3.875, 2.8125}, {3.6875, 7.4375}},
     {{1.25, 1.625}, {2.0, 3.75}, {4.125, 2.875}}},
    80,
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"}};

int Yolov3PostProcessMethod::Init(const std::string &cfg_path) {
  DnnPostProcessMethod::Init(cfg_path);

  score_threshold_ = config_.GetFloatValue("score_threshold", score_threshold_);
  nms_threshold_ = config_.GetFloatValue("nms_threshold", nms_threshold_);
  nms_top_k_ = config_.GetIntValue("nms_top_k", nms_top_k_);

  basic_pyramid_image_height_ =
      config_.GetIntValue("basic_pyramid_image_height");
  basic_pyramid_image_width_ =
      config_.GetIntValue("basic_pyramid_image_width");
  return 0;
}

int Yolov3PostProcessMethod::ParseDnnResult(
    DnnAsyncData &dnn_result,
    std::vector<BaseDataPtr> &frame_result) {
  LOGD << "Yolov3PostProcessMethod ParseDnnResult";

  // 检测对应一次预测
  HOBOT_CHECK(dnn_result.output_tensors.size() == 1);
  auto &output_tensors = dnn_result.output_tensors[0];

  static int model_input_height, model_input_width;  // 模型输入大小
  static std::once_flag flag;
  std::call_once(flag, [&dnn_result]() {
    int ret = HB_BPU_getHW(dnn_result.dnn_model->bpu_model.inputs[0].data_type,
                           &dnn_result.dnn_model->bpu_model.inputs[0].shape,
                           &model_input_height, &model_input_width);
    if (ret != 0) {
      LOGE << "Error getting model input size";
    }
  });

  std::vector<Detection> dets;
  int out_layer = dnn_result.dnn_model->bpu_model.output_num;
  LOGD << "yolov3 model output layer: " << out_layer;
  for (int i = 0; i < out_layer; ++i) {
    // flush cache
    auto &tensor = output_tensors[i];
    HB_SYS_flushMemCache(&(tensor.data),
                         HB_SYS_MEM_CACHE_INVALIDATE);
    auto *data = reinterpret_cast<float *>(tensor.data.virAddr);
    int num_classes = default_yolo3_config.class_num;
    int stride = default_yolo3_config.strides[i];
    int num_pred = default_yolo3_config.class_num + 4 + 1;

    std::vector<float> class_pred(default_yolo3_config.class_num, 0.0);
    std::vector<std::pair<double, double>> &anchors =
        default_yolo3_config.anchors_table[i];

    double h_ratio = basic_pyramid_image_height_ * 1.0 /
                     dnn_result.src_image_height;
    double w_ratio = basic_pyramid_image_width_ * 1.0 /
                     dnn_result.src_image_width;

    int height, width;
    int ret = HB_BPU_getHW(
        tensor.data_type, &tensor.data_shape, &height, &width);
    if (ret != 0) {
      LOGE << "HB_BPU_getHW failed";
    }
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (size_t k = 0; k < anchors.size(); k++) {
          double anchor_x = anchors[k].first;
          double anchor_y = anchors[k].second;
          float *cur_data = data + k * num_pred;
          float objness = cur_data[4];
          for (int index = 0; index < num_classes; ++index) {
            class_pred[index] = cur_data[5 + index];
          }

          // argmax
          float id = std::distance(
              class_pred.begin(),
              std::max_element(class_pred.begin(), class_pred.end()));
          double x1 = 1 / (1 + std::exp(-objness)) * 1;
          double x2 = 1 / (1 + std::exp(-class_pred[id]));
          double confidence = x1 * x2;

          if (confidence < score_threshold_) {
            continue;
          }

          float center_x = cur_data[0];
          float center_y = cur_data[1];
          float scale_x = cur_data[2];
          float scale_y = cur_data[3];

          double box_center_x =
              ((1.0 / (1.0 + std::exp(-center_x))) + w) * stride;
          double box_center_y =
              ((1.0 / (1.0 + std::exp(-center_y))) + h) * stride;

          double box_scale_x = std::exp(scale_x) * anchor_x * stride;
          double box_scale_y = std::exp(scale_y) * anchor_y * stride;

          double xmin = (box_center_x - box_scale_x / 2.0);
          double ymin = (box_center_y - box_scale_y / 2.0);
          double xmax = (box_center_x + box_scale_x / 2.0);
          double ymax = (box_center_y + box_scale_y / 2.0);

          // 预处理是将金字塔数据(960x540)，padding到960x960后,再缩放到416x416
          // 将坐标从模型输入大小(416x416)，映射到(960x960)
          float factor = basic_pyramid_image_width_ * 1.0 / model_input_width;
          xmin *= factor;
          xmax *= factor;
          ymin *= factor;
          ymax *= factor;
          // 将坐标压缩在图像范围内(960x540)
          ymin = std::min(
              ymin, static_cast<double>(basic_pyramid_image_height_) - 1);
          ymax = std::min(
              ymax, static_cast<double>(basic_pyramid_image_height_) - 1);

          // 将坐标从金字塔大小(960x540)，映射到原图大小(1920 x 1080)
          double xmin_org = xmin / w_ratio;
          double xmax_org = xmax / w_ratio;
          double ymin_org = ymin / h_ratio;
          double ymax_org = ymax / h_ratio;

          if (xmin_org > xmax_org || ymin_org > ymax_org) {
            continue;
          }

          xmin_org = std::max(xmin_org, 0.0);
          xmax_org = std::min(xmax_org, dnn_result.src_image_width - 1.0);
          ymin_org = std::max(ymin_org, 0.0);
          ymax_org = std::min(ymax_org, dnn_result.src_image_height - 1.0);

          hobot::vision::BBox bbox(
              xmin_org, ymin_org, xmax_org, ymax_org,
              confidence, static_cast<int>(id),
              default_yolo3_config.class_names[static_cast<int>(id)].c_str());
          dets.push_back(Detection(bbox));
        }
        data = data + num_pred * anchors.size();
      }
    }
  }
  std::vector<Detection> det_result;
  nms(dets, nms_threshold_, nms_top_k_, det_result, false);

  // Detection转换BaseData
  auto xstream_det_result = std::make_shared<xstream::BaseDataVector>();
  for (size_t i = 0; i < det_result.size(); ++i) {
    const auto &box = det_result[i].bbox;
    LOGD << box << ", id: " << box.id
         << ", category_name: " << box.category_name;
    auto xstream_box =
        std::make_shared<xstream::XStreamData<hobot::vision::BBox>>();
    xstream_box->value = std::move(box);
    xstream_det_result->datas_.push_back(xstream_box);
  }
  frame_result.push_back(xstream_det_result);

  return 0;
}

void Yolov3PostProcessMethod::nms(
    std::vector<Detection> &input,
    float iou_threshold,
    int top_k,
    std::vector<Detection> &result,
    bool suppress) {
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<Detection>());
  if (input.size() > 400) {
    input.resize(400);
  }

  std::vector<bool> skip(input.size(), false);

  // pre-calculate boxes area
  std::vector<float> areas;
  areas.reserve(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    float width = input[i].bbox.Width();
    float height = input[i].bbox.Height();
    areas.push_back(width * height);
  }

  int count = 0;
  for (size_t i = 0; count < top_k && i < skip.size(); i++) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      if (suppress == false) {
        if (input[i].bbox.id != input[j].bbox.id) {
          continue;
        }
      }

      // intersection area
      float xx1 = std::max(input[i].bbox.x1, input[j].bbox.x1);
      float yy1 = std::max(input[i].bbox.y1, input[j].bbox.y1);
      float xx2 = std::min(input[i].bbox.x2, input[j].bbox.x2);
      float yy2 = std::min(input[i].bbox.y2, input[j].bbox.y2);

      if (xx2 > xx1 && yy2 > yy1) {
        float area_intersection = (xx2 - xx1) * (yy2 - yy1);
        float iou_ratio =
            area_intersection / (areas[j] + areas[i] - area_intersection);
        if (iou_ratio > iou_threshold) {
          skip[j] = true;
        }
      }
    }
    result.push_back(input[i]);
  }
}

}  // namespace xstream
