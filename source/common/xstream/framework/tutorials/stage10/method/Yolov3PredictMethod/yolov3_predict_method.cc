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
      const InputParamPtr param,
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

  // 全图检测对应一次预测
  input_tensors.resize(1);
  output_tensors.resize(1);

#ifdef X2
  auto input_img = pyramid_image->img.down_scale[pyramid_layer_];
#endif
#ifdef X3
  auto input_img = pyramid_image->down_scale[pyramid_layer_];
#endif
  // 期望金字塔图像宽高都大于模型输入，避免padding影响算法效果
  if (input_img.height < model_input_height_ ||
      input_img.width < model_input_width_) {
    LOGE << "pyramid image size smaller than model_input size, "
         << "please check pyramid_layer";
    return -1;
  }

  // 1. alloc input_tensors
  int ret = AllocInputTensor(input_tensors[0]);
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

  // 模型输入大小：416 x 416
  // 以1080p为例，金字塔第0层1920x1080，第4层960x540
  // vio配置目标层pyramid_layer_：取第4层数据，padding到960x960，再resize到416x416
  // desired: target_pym_layer_height = 540, target_pym_layer_width = 960

  // prepare 960x960 BPU_TENSOR
  // 注意BPU_TENSOR需要释放
  BPU_TENSOR_S pre_resize_tensor;
  {
    pre_resize_tensor.data_type = BPU_TYPE_IMG_NV12_SEPARATE;
    int h_idx, w_idx, c_idx;
    HB_BPU_getHWCIndex(
        pre_resize_tensor.data_type, nullptr, &h_idx, &w_idx, &c_idx);
    pre_resize_tensor.data_shape.ndim = 4;
    pre_resize_tensor.data_shape.d[0] = 1;
    // 金字塔图像width>height, padding到width
    pre_resize_tensor.data_shape.d[h_idx] = input_img.width;
    pre_resize_tensor.data_shape.d[w_idx] = input_img.width;
    pre_resize_tensor.data_shape.d[c_idx] = 3;
    pre_resize_tensor.aligned_shape = pre_resize_tensor.data_shape;

    // alloc bpu-mem
    // 数据类型是nv12图像，y和uv分量分开alloc和存储
    int y_length = input_img.width * input_img.width;
    int uv_length = y_length >> 1;
    ret = HB_SYS_bpuMemAlloc(
        "in_data0", y_length, true, &pre_resize_tensor.data);
    if (ret != 0) {
      LOGE << "bpu alloc mem failed: " << HB_BPU_getErrorName(ret);
      return -1;
    }
    ret = HB_SYS_bpuMemAlloc(
        "in_data1", uv_length, true, &pre_resize_tensor.data_ext);
    if (ret != 0) {
      LOGE << "bpu alloc mem failed: " << HB_BPU_getErrorName(ret);
      // release alloced mem
      HB_SYS_bpuMemFree(&pre_resize_tensor.data);
      return -1;
    }
    // Copy y data to data0
    uint8_t *y = reinterpret_cast<uint8_t *>(pre_resize_tensor.data.virAddr);
    uint8_t *src_y = reinterpret_cast<uint8_t *>(input_img.y_vaddr);
    memcpy(y, src_y, input_img.width * input_img.height);
    // 将padding的底部填充0
    memset(y + input_img.width * input_img.height, 0,
           input_img.width * (input_img.width - input_img.height));
    HB_SYS_flushMemCache(&pre_resize_tensor.data, HB_SYS_MEM_CACHE_CLEAN);
    // Copy uv data to data_ext
    uint8_t *uv = reinterpret_cast<uint8_t *>(
        pre_resize_tensor.data_ext.virAddr);
    uint8_t *src_uv = reinterpret_cast<uint8_t *>(input_img.c_vaddr);
    int uv_height = input_img.height >> 1;
    memcpy(uv, src_uv, input_img.width * uv_height);
    // 将padding的底部填充0
    memset(uv + input_img.width * uv_height, 0,
           input_img.width * (input_img.width - input_img.height) >> 1);
    HB_SYS_flushMemCache(&pre_resize_tensor.data_ext, HB_SYS_MEM_CACHE_CLEAN);
  }

  // resize to 416x416
  BPU_RESIZE_CTRL_S ctrl_param = {
      BPU_RESIZE_TYPE_BILINEAR,
      BPU_TYPE_IMG_NV12_SEPARATE,
      -1};
  HOBOT_CHECK(input_tensors[0].size() == 1);
  ret = HB_BPU_resize(&pre_resize_tensor,
                      &input_tensors[0][0],
                      &ctrl_param);
  HB_SYS_bpuMemFree(&pre_resize_tensor.data);
  HB_SYS_bpuMemFree(&pre_resize_tensor.data_ext);
  if (ret != 0) {
    LOGE << "resize nv12 failed";
    return -1;
  }
  return 0;
}
}  // namespace xstream
