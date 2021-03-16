/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: APAFullimgMultitaskPredictMethod.cpp
 * @Brief: definition of the APAFullimgMultitaskPredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2021-01-08 14:27:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2021-01-08 19:07:12
 */

#include "APAFullimgMultitaskPredictMethod/APAFullimgMultitaskPredictMethod.h"
#include <string>
#include <vector>
#include "hobotlog/hobotlog.hpp"
#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/imgproc.hpp"

using hobot::vision::PymImageFrame;
using hobot::vision::CVImageFrame;

namespace xstream {

int APAFullimgMultitaskPredictMethod::Init(const std::string &cfg_path) {
  DnnPredictMethod::Init(cfg_path);
  pyramid_layer_ = config_["pyramid_layer"].isInt() ?
                   config_["pyramid_layer"].asInt() : pyramid_layer_;
  return 0;
}

int APAFullimgMultitaskPredictMethod::GetSrcImageSize(
    const std::vector<BaseDataPtr> &input, int &src_image_height,
    int &src_image_width) {
  HOBOT_CHECK(input.size() == 1);  // image
  auto xstream_img = std::static_pointer_cast<
      XStreamData<std::shared_ptr<hobot::vision::ImageFrame>>>(input[0]);

  std::string img_type = xstream_img->value->type;
  if (img_type == "PymImageFrame") {
    auto pyramid_image =
        std::static_pointer_cast<PymImageFrame>(xstream_img->value);
#ifdef X2
    src_image_height = pyramid_image->img.src_img.height;
    src_image_width = pyramid_image->img.src_img.width;
#endif

#ifdef X3
    src_image_height = pyramid_image->down_scale[0].height;
    src_image_width = pyramid_image->down_scale[0].width;
#endif
  } else if (img_type == "CVImageFrame") {
    auto cv_image = std::static_pointer_cast<CVImageFrame>(xstream_img->value);
    src_image_height = cv_image->Height();
    src_image_width = cv_image->Width();
  } else {
    LOGE << "not support " << img_type;
    return -1;
  }
  LOGD << "src image height: " << src_image_height
       << ", src image width: " << src_image_width;
  return 0;
}

int APAFullimgMultitaskPredictMethod::PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const InputParamPtr param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors) {
  LOGD << "APAFullimgMultitaskPredictMethod PrepareInputData";
  HOBOT_CHECK(input.size() == 1);  // image

  auto xstream_img = std::static_pointer_cast<XStreamData<
      std::shared_ptr<hobot::vision::ImageFrame>>>(input[0]);       // image

  std::string img_type = xstream_img->value->type;
  HOBOT_CHECK(img_type == "PymImageFrame" ||
              img_type == "CVImageFrame") << "not support " << img_type;

  // 全图检测对应一次
  input_tensors.resize(1);
  output_tensors.resize(1);

  uint8_t *input_y_data, *input_uv_data;
  if (img_type == "PymImageFrame") {
    auto pyramid_image = std::static_pointer_cast<
        hobot::vision::PymImageFrame>(xstream_img->value);

    int target_pym_layer_height = 0;
    int target_pym_layer_width = 0;
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
    // 模型输入大小：1280x704，金字塔选层大小：1280x720
    // 取前704行做输入
    if (target_pym_layer_height != 720 ||
        target_pym_layer_width != 1280) {
      LOGE << "input image size not match, please check pyramid_layer";
      return -1;
    }

#ifdef X2
    input_y_data = reinterpret_cast<uint8_t *>(
        pyramid_image->img.down_scale[pyramid_layer_].y_vaddr);
    input_uv_data = reinterpret_cast<uint8_t *>(
        pyramid_image->img.down_scale[pyramid_layer_].c_vaddr);
#endif
#ifdef X3
    input_y_data = reinterpret_cast<uint8_t *>(
        pyramid_image->down_scale[pyramid_layer_].y_vaddr);
    input_uv_data = reinterpret_cast<uint8_t *>(
        pyramid_image->down_scale[pyramid_layer_].c_vaddr);
#endif
  } else {
    auto cv_image = std::static_pointer_cast<
        hobot::vision::CVImageFrame>(xstream_img->value);
    // support nv12, bgr
    HOBOT_CHECK(cv_image->pixel_format ==
                HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawNV12 ||
                cv_image->pixel_format ==
                HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawBGR);
    int src_img_height = cv_image->Height();
    int src_img_width = cv_image->Width();
    auto &img_mat = cv_image->img;
    if (src_img_height != 720 ||
        src_img_width != 1280) {
      LOGE << "input image size not match(720x1280), please check it";
      return -1;
    }
    // nv12
    if (cv_image->pixel_format ==
        HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawNV12) {
      // 取前704行做输入
      // 设置input_y_data, input_uv_data
      input_y_data = img_mat.ptr<uint8_t>();
      input_uv_data = input_y_data + src_img_height * src_img_width;
    } else {
      // bgr_to_nv12
      cv::Mat yuv_mat;
      cv::cvtColor(img_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
      uint8_t *yuv = yuv_mat.ptr<uint8_t>();
      int uv_height = src_img_height / 2;
      int uv_width = src_img_width / 2;
      int y_size = uv_height * uv_width * 4;

      static uint8_t nv12[720 * 1280 * 3 / 2];
      memcpy(nv12, yuv, y_size);

      int uv_stride = uv_width * uv_height;
      uint8_t *uv_data = nv12 + y_size;
      for (int i = 0; i < uv_stride; ++i) {
        *(uv_data++) = *(yuv + y_size + i);
        *(uv_data++) = *(yuv + y_size + +uv_stride + i);
      }
      // 设置input_y_data, input_uv_data
      input_y_data = nv12;
      input_uv_data = nv12 + y_size;
    }
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
  // 3. copy data to input_tensors
  HOBOT_CHECK(input_tensors[0].size() == 1);  // 1层输入
  HOBOT_CHECK(input_tensors[0][0].data_type == BPU_TYPE_IMG_NV12_SEPARATE);
  {
    BPU_TENSOR_S &tensor = input_tensors[0][0];
    int height = tensor.data_shape.d[input_h_idx_];
    int width = tensor.data_shape.d[input_w_idx_];
    int stride = tensor.aligned_shape.d[input_w_idx_];

    // Copy y data to data0
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
