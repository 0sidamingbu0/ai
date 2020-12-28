/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: yolov3_predict_method.cc
 * @Brief: definition of the Yolov3PredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 11:12:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 21:21:33
 */

#include "Yolov3PredictMethod/yolov3_predict_method.h"
#include <string>
#include <vector>
#include <memory>
#include "opencv2/highgui/highgui.hpp"
#include "hobotxstream/profiler.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "horizon/vision_type/vision_type.hpp"
#include "hobotxstream/image_tools.h"
#include "hobotlog/hobotlog.hpp"
#include "DnnAsyncData.h"

#ifdef X3
#include "./bpu_predict_x3.h"
#endif
namespace xstream {

int Yolov3PredictMethod::Init(const std::string &cfg_path) {
  DnnPredictMethod::Init(cfg_path);
  pyramid_layer_ = config_["pyramid_layer"].isInt() ?
                   config_["pyramid_layer"].asInt() : pyramid_layer_;
  return 0;
}

int Yolov3PredictMethod::GetSrcImageSize(
    const std::vector<BaseDataPtr> &input, int &src_image_height,
    int &src_image_width) {
  HOBOT_CHECK(input.size() == 1);  // image
  auto xstream_img = std::static_pointer_cast<
      XStreamData<std::shared_ptr<hobot::vision::ImageFrame>>>(input[0]);

  std::string img_type = xstream_img->value->type;
  if (img_type == "PymImageFrame") {
    auto pyramid_image = std::static_pointer_cast<
        hobot::vision::PymImageFrame>(xstream_img->value);
#ifdef X2
    src_image_height = pyramid_image->img.src_img.height;
    src_image_width = pyramid_image->img.src_img.width;
#endif

#ifdef X3
    src_image_height = pyramid_image->down_scale[0].height;
    src_image_width = pyramid_image->down_scale[0].width;
#endif
  } else {
    LOGE << "not support " << img_type;
    return -1;
  }
  LOGD << "src image height: " << src_image_height
       << ", src image width: " << src_image_width;
  return 0;
}

int Yolov3PredictMethod::PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors) {
  LOGD << "Yolov3PredictMethod PrepareInputData";
  HOBOT_CHECK(input.size() == 1);  // image

  auto xstream_img = std::static_pointer_cast<XStreamData<
      std::shared_ptr<hobot::vision::ImageFrame>>>(input[0]);       // image

  std::string img_type = xstream_img->value->type;
  HOBOT_CHECK(img_type == "PymImageFrame") << "not support " << img_type;

  auto pyramid_image = std::static_pointer_cast<
      hobot::vision::PymImageFrame>(xstream_img->value);
  // pym第0层大小
  // const int src_width = pyramid_image->Width();
  // const int src_height = pyramid_image->Height();

  // 全图检测对应一次预测
  input_tensors.resize(1);
  output_tensors.resize(1);

  int target_pym_layer_height = 0;
  int target_pym_layer_width = 0;

  // 模型输入大小：416 x 416
  // 以1080p为例，金字塔第0层1920x1080，第4层960x540，第8层480x270
  // vio配置目标层pyramid_layer_：从第4层抠取图像中间的416x416
  // desired: target_pym_layer_height = 416, target_pym_layer_width = 416
#ifdef X2
  target_pym_layer_height =
      pyramid_image->img.down_scale[pyramid_layer_].height;
  target_pym_layer_width =
      pyramid_image->img.down_scale[pyramid_layer_].width;
#endif
#ifdef X3
  target_pym_layer_height = pyramid_image->down_scale[pyramid_layer_].height;
  target_pym_layer_width = pyramid_image->down_scale[pyramid_layer_].width;
#endif

  int ret = 0;

  // check pyramid image size
  if (target_pym_layer_height != model_input_height_ ||
      target_pym_layer_width != model_input_width_) {
    LOGE << "pyramid image size not equal to model input size, "
        << "check vio_config and pyramid_layer: " << pyramid_layer_;
    return -1;
  }

#ifdef X2
  auto input_img = pyramid_image->img.down_scale[pyramid_layer_];
#endif
#ifdef X3
  auto input_img = pyramid_image->down_scale[pyramid_layer_];
#endif

  // 1. alloc input_tensors
  ret = AllocInputTensor(input_tensors[0]);
  if (ret != 0) {
  LOGE << "Alloc InputTensor failed!";
  return -1;
  }
  // 2. alloc output_tensors
  ret = AllocOutputTensor(output_tensors[0]);
  if (ret != 0) {
    LOGE << "Alloc OutputTensor failed!";
    return -1;
  }

  // 3. copy data to input_tensors
  HOBOT_CHECK(input_tensors[0].size() == 1);  // 1层输入
  // 与模型有关，需要根据模型信息处理,此模型输入数据类型nv12
  HOBOT_CHECK(input_tensors[0][0].data_type == BPU_TYPE_IMG_NV12_SEPARATE);
  {
    uint8_t *input_y_data, *input_uv_data;

    BPU_TENSOR_S &tensor = input_tensors[0][0];
    int height = tensor.data_shape.d[input_h_idx_];
    int width = tensor.data_shape.d[input_w_idx_];
    int stride = tensor.aligned_shape.d[input_w_idx_];
#ifdef X2
    input_y_data = reinterpret_cast<uint8_t *>(input_img.y_vaddr);
    input_uv_data = reinterpret_cast<uint8_t *>(input_img.c_vaddr);
#endif
#ifdef X3
    input_y_data = reinterpret_cast<uint8_t *>(input_img.y_vaddr);
    input_uv_data = reinterpret_cast<uint8_t *>(input_img.c_vaddr);
#endif

    // copy y data to data0
    uint8_t *y = reinterpret_cast<uint8_t *>(tensor.data.virAddr);
    for (int h = 0; h < height; ++h) {
      auto *raw = y + h * stride;
      memcpy(raw, input_y_data, width);
      input_y_data += width;
    }
    HB_SYS_flushMemCache(&tensor.data, HB_SYS_MEM_CACHE_CLEAN);

    // Copy uv data to data_ext
    uint8_t *uv = reinterpret_cast<uint8_t *>(tensor.data_ext.virAddr);
    int uv_height = height / 2;
    for (int i = 0; i < uv_height; ++i) {
      auto *raw = uv + i * stride;
      memcpy(raw, input_uv_data, width);
      input_uv_data += width;
    }
    HB_SYS_flushMemCache(&tensor.data_ext, HB_SYS_MEM_CACHE_CLEAN);
  }

  return 0;
}
}  // namespace xstream
