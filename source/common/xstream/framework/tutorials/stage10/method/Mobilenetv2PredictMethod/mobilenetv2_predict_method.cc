/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: mobilenetv2_predict_method.cc
 * @Brief: definition of the Mobilenetv2PredictMethod
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-12-23 11:12:18
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-12-23 21:21:33
 */

#include "Mobilenetv2PredictMethod/mobilenetv2_predict_method.h"
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

int Mobilenetv2PredictMethod::Init(const std::string &cfg_path) {
  DnnPredictMethod::Init(cfg_path);
  return 0;
}

int Mobilenetv2PredictMethod::PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors) {
  LOGD << "Mobilenetv2PredictMethod PrepareInputData";
  HOBOT_CHECK(input.size() == 2);  // image, roi

  auto xstream_img = std::static_pointer_cast<XStreamData<
      std::shared_ptr<hobot::vision::ImageFrame>>>(input[0]);       // image
  auto rois = std::static_pointer_cast<BaseDataVector>(input[1]);   // body_box

  std::string img_type = xstream_img->value->type;
  HOBOT_CHECK(img_type == "PymImageFrame") << "not support " << img_type;

  auto pyramid_image = std::static_pointer_cast<
      hobot::vision::PymImageFrame>(xstream_img->value);
  const int width = pyramid_image->Width();  // 原图大小
  const int height = pyramid_image->Height();

  int64_t box_num = rois->datas_.size();
  // 一个roi对应一次预测
  input_tensors.resize(box_num);
  output_tensors.resize(box_num);

  for (int32_t roi_idx = 0; roi_idx < box_num; roi_idx++) {
    auto roi = std::static_pointer_cast<XStreamData<hobot::vision::BBox>>(
        rois->datas_[roi_idx]);
    if (roi->state_ != xstream::DataState::VALID) {
      continue;
    }
    int ret = 0;
    // model input size: 224x224x3
    // 1. 扣取金字塔roi图像，并resize到模型输入大小
    // 1.1 准备原图数据，封装tensor
    BPU_TENSOR_S nv12_tensor;
    nv12_tensor.data_type = BPU_TYPE_IMG_NV12_SEPARATE;
    int h_idx, w_idx, c_idx;
    HB_BPU_getHWCIndex(nv12_tensor.data_type, nullptr, &h_idx, &w_idx, &c_idx);
    {
      nv12_tensor.data_shape.ndim = 4;
      nv12_tensor.data_shape.d[0] = 1;
      nv12_tensor.data_shape.d[h_idx] = height;
      nv12_tensor.data_shape.d[w_idx] = width;
      nv12_tensor.data_shape.d[c_idx] = 3;
      nv12_tensor.aligned_shape = nv12_tensor.data_shape;

      // Copy y data to data0
      int pym_layer = pyramid_image->pym_layer;
      #ifdef X2
      nv12_tensor.data.virAddr = reinterpret_cast<void*>(
          pyramid_image->img.down_scale[pym_layer].y_vaddr);
      nv12_tensor.data.phyAddr =
          pyramid_image->img.down_scale[pym_layer].y_paddr;
      nv12_tensor.data.memSize =  height * width;
      // Copy uv data to data_ext
      nv12_tensor.data_ext.virAddr = reinterpret_cast<void*>(
          pyramid_image->img.down_scale[pym_layer].c_vaddr);
      nv12_tensor.data_ext.phyAddr =
          pyramid_image->img.down_scale[pym_layer].c_paddr;
      nv12_tensor.data_ext.memSize =  (height + 1) / 2 * width;
      #endif
      #ifdef X3
      nv12_tensor.data.virAddr = reinterpret_cast<void*>(
          pyramid_image->down_scale[pym_layer].y_vaddr);
      nv12_tensor.data.phyAddr = pyramid_image->down_scale[pym_layer].y_paddr;
      nv12_tensor.data.memSize =  height * width;
      // Copy uv data to data_ext
      nv12_tensor.data_ext.virAddr = reinterpret_cast<void*>(
          pyramid_image->down_scale[pym_layer].c_vaddr);
      nv12_tensor.data_ext.phyAddr =
          pyramid_image->down_scale[pym_layer].c_paddr;
      nv12_tensor.data_ext.memSize =  (height + 1) / 2 * width;
      #endif
    }

    // 2. 申请input/output tensor
    ret = AllocInputTensor(input_tensors[roi_idx]);
    if (ret != 0) {
      LOGE << "ROI index: " << roi_idx << ", Alloc InputTensor failed!";
      continue;
    }
    ret = AllocOutputTensor(output_tensors[roi_idx]);
    if (ret != 0) {
      LOGE << "ROI index: " << roi_idx << ", Alloc OutputTensor failed!";
      continue;
    }

    // 3. crop & resize nv12数据到input tensor
    HOBOT_CHECK(input_tensors[roi_idx].size() == 1)
        << "expected input num: 1";
    HOBOT_CHECK(input_tensors[roi_idx][0].data_type ==
                BPU_TYPE_IMG_NV12_SEPARATE)
        << "expected input tensor type: nv12";

    // roi && resize数据, 封装tensor
    BPU_ROI_S input_roi = {
        static_cast<int>(roi->value.x1), static_cast<int>(roi->value.y1),
        static_cast<int>(roi->value.x2), static_cast<int>(roi->value.y2)};
    BPU_RESIZE_CTRL_S ctrl_param = {
        BPU_RESIZE_TYPE_BILINEAR,
        BPU_TYPE_IMG_NV12_SEPARATE,
        -1};
    ret = HB_BPU_cropAndResize(
        &nv12_tensor, &input_roi, &input_tensors[roi_idx][0], &ctrl_param);
    if (ret != 0) {
      LOGE << "box index: " << roi_idx
           << ", HB_BPU_cropAndResize failed, ret: " << ret << ": "
           << HB_BPU_getErrorName(ret);
      LOGE << "input_roi (x1,y1,x2,y2): "
           << input_roi.x1 << ", " << input_roi.y1 << ", "
           << input_roi.x2 << ", " << input_roi.y2;
      continue;
    }
  }
  return 0;
}
}  // namespace xstream
