/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: DetectPredictor.cc
 * @Brief: definition of the DetectPredictor
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-08-19 14:28:17
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-08-19 16:19:08
 */

#include "PredictMethod/Predictors/CropDetectPredictor.h"
#include <vector>
#include <memory>
#include <xstream/vision_type/include/horizon/vision_type/vision_error.h>
#include <xstream/imagetools/include/hobotxstream/image_tools.h>
#include <queue>
#include "hobotxstream/profiler.h"
#include "hobotxsdk/xstream_data.h"
#include "hobotlog/hobotlog.hpp"
#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/imgproc.hpp"

namespace xstream {

using BPU_TENSOR_S_LIST_PTR = std::shared_ptr<std::vector<BPU_TENSOR_S>>;
using BPU_TENSOR_S_LIST_PTR_LIST = std::vector<BPU_TENSOR_S_LIST_PTR>;

std::vector<BaseDataPtr> CropDetectPredictor::Do(
    const std::vector<BaseDataPtr> &input,
    const xstream::InputParamPtr &param) {
  std::vector<BaseDataPtr> output;
  output = RunSingleFrame(input);
  return output;
}

int CropDetectPredictor::CropPadAndResizeRoi(
    hobot::vision::BBox *norm_box,
    uint8_t *pym_src_y_data, uint8_t *pym_src_uv_data,
    int pym_src_w, int pym_src_h,
    int pym_src_y_size, int pym_src_uv_size,
    int pym_src_y_stride, int pym_src_uv_stride,
    uint8_t **img_data, HobotXStreamImageToolsResizeInfo *resize_info) {

  uint8_t *pym_dst_data = nullptr;
  int pym_dst_size, pym_dst_w, pym_dst_h, pym_dst_y_stride, pym_dst_uv_stride;
  const uint8_t *input_yuv_data[3] = {pym_src_y_data, pym_src_uv_data, nullptr};
  const int input_yuv_size[3] = {pym_src_y_size, pym_src_uv_size, 0};

  int ret = 0;
  {
    RUN_PROCESS_TIME_PROFILER("Run Detect CropYUVImage");
    RUN_FPS_PROFILER("Run Detect CropYUVImage");

    ret = HobotXStreamCropYUVImage(
        input_yuv_data, input_yuv_size, pym_src_w, pym_src_h, pym_src_y_stride,
        pym_src_uv_stride, IMAGE_TOOLS_RAW_YUV_NV12, norm_box->x1, norm_box->y1,
        norm_box->x2 - 1, norm_box->y2 - 1, &pym_dst_data, &pym_dst_size,
        &pym_dst_w, &pym_dst_h, &pym_dst_y_stride, &pym_dst_uv_stride);
    HOBOT_CHECK(ret == 0)
    << "crop img failed"
    << ", pym_src_y_size:" << pym_src_y_size << ", pym_src_w:" << pym_src_w
    << ", pym_src_h:" << pym_src_h << ", pym_src_y_stride:" << pym_src_y_stride
    << ", pym_src_uv_stride:" << pym_src_uv_stride;
  }
  pym_src_y_data = pym_dst_data;
  pym_src_y_size = pym_dst_size;
  pym_src_uv_data = nullptr;
  pym_src_uv_size = 0;
  pym_src_w = pym_dst_w;
  pym_src_h = pym_dst_h;
  pym_src_y_stride = pym_dst_y_stride;
  pym_src_uv_stride = pym_dst_uv_stride;

  if (resize_type_ == BPU_RESIZE) {
    RUN_PROCESS_TIME_PROFILER("Run Detect PadYUVImage");
    RUN_FPS_PROFILER("Run Detect PadYUVImage");
    resize_info->src_height_ = pym_dst_h;
    resize_info->src_width_ = pym_dst_w;
    CalcResizeInfo(resize_info);
    const uint8_t padding_value[3] = {0, 128, 128};

    int padding_right =
        resize_info->padding_right_ / resize_info->width_ratio_;
    padding_right &= ~1;
    int padding_bottom =
        resize_info->padding_bottom_ / resize_info->height_ratio_;
    padding_bottom &= ~1;

    ret = HobotXStreamPadImage(
        pym_src_y_data, pym_src_y_size,
        pym_src_w, pym_src_h,
        pym_src_y_stride, pym_src_uv_stride,
        IMAGE_TOOLS_RAW_YUV_NV12,
        0, padding_right, 0, padding_bottom,
        padding_value,
        &pym_dst_data, &pym_dst_size,
        &pym_dst_w, &pym_dst_h,
        &pym_dst_y_stride, &pym_dst_uv_stride);
  } else if (resize_type_ == CPU_RESIZE) {
    RUN_PROCESS_TIME_PROFILER("Run Detect ResizeYUVImage");
    RUN_FPS_PROFILER("Run Detect ResizeYUVImage");
    //  The aspect ratio of original image's width to hight
    //  can be held when 'fix_aspect_ratio' is set to '1',
    //  where the original image be padded with black value.
    //  Hence, there is no need to call HobotXStreamPadImage
    //  function before HobotXStreamResizeImage.
    ret = HobotXStreamResizeImage(
        pym_src_y_data, pym_src_y_size,
        pym_src_w, pym_src_h,
        pym_src_y_stride, pym_src_uv_stride,
        IMAGE_TOOLS_RAW_YUV_NV12,
        1,
        model_input_width_, model_input_height_,
        &pym_dst_data, &pym_dst_size, &pym_dst_y_stride, &pym_dst_uv_stride,
        resize_info);
  } else {
    HOBOT_CHECK(false) << "resize type undefined";
  }

//  static uint32_t id = 0;
//  std::string output_file_path = std::to_string(id++) + ".yuv";
//  std::ofstream outfile(output_file_path, std::ios::out | std::ios::binary);
//  outfile.write(reinterpret_cast<const char *>(pym_dst_data), pym_dst_size);
//  outfile.close();

  //  free the crop image
  HobotXStreamFreeImage(pym_src_y_data);
  HOBOT_CHECK(ret == 0)
  << "resize img failed, ret: " << ret
  << ", pym_src_y_size:" << pym_src_y_size << ", pym_src_w:" << pym_src_w
  << ", pym_src_h:" << pym_src_h << ", pym_src_y_stride:" << pym_src_y_stride
  << ", pym_src_uv_stride:" << pym_src_uv_stride;

  *img_data = pym_dst_data;
  return kHorizonVisionSuccess;
}

void CropDetectPredictor::CalcResizeInfo(
    HobotXStreamImageToolsResizeInfo *resize_info) {
  resize_info->dst_width_ = model_input_width_;
  resize_info->dst_height_ = model_input_height_;
  float width_ratio = static_cast<float>(model_input_width_)
      / resize_info->src_width_;
  float height_ratio = static_cast<float>(model_input_height_)
      / resize_info->src_height_;

  // fix_aspect_ratio
  float ratio = width_ratio;
  if (ratio > height_ratio) {
    ratio = height_ratio;
  }
  resize_info->padding_right_ = model_input_width_ / ratio
      - resize_info->src_width_;
  resize_info->padding_bottom_ = model_input_height_ / ratio
      - resize_info->src_height_;
  resize_info->width_ratio_ = 1;
  resize_info->height_ratio_ = 1;
}

int CropDetectPredictor::GetPymScaleInfo(
    const std::shared_ptr<hobot::vision::PymImageFrame>& pyramid,
    hobot::vision::BBox *norm_box,
    uint8_t* &pym_src_y_data, uint8_t* &pym_src_uv_data,
    int &pym_src_w, int &pym_src_h,
    int &pym_src_y_size, int &pym_src_uv_size,
    int &pym_src_y_stride, int &pym_src_uv_stride, float &scale) {
  auto box_width = norm_box->Width();
  auto box_height = norm_box->Height();
  auto min_cost_scale = 1, minareadiff = INT_MAX;
  int target_area = model_input_width_ * model_input_height_;
  int box_area =  box_height * box_width;
  for (int i = min_cost_scale; i <= 3; i++) {
    int areadiff = std::abs(box_area - target_area);
    if (minareadiff > areadiff) {
      minareadiff = areadiff;
      min_cost_scale = i;
    }
    box_area >>= 2;
  }
  int pyramid_scale = 1 << (min_cost_scale - 1);
  pyramid_scale = (pyramid_scale == 1 ? 0 : pyramid_scale) * 2;
  assert(pyramid_scale >= 0 && pyramid_scale < DOWN_SCALE_MAX);
  auto &pyramid_down_scale = pyramid->down_scale[pyramid_scale];
  pym_src_y_data = reinterpret_cast<uint8_t *>(
      pyramid_down_scale.y_vaddr);
  pym_src_uv_data = reinterpret_cast<uint8_t *>(
      pyramid_down_scale.c_vaddr);

  pym_src_w = pyramid_down_scale.width;
  pym_src_h = pyramid_down_scale.height;

  pym_src_y_stride = pyramid_down_scale.stride;
  pym_src_uv_stride = pyramid_down_scale.stride;

  pym_src_y_size = pym_src_y_stride * pym_src_h;
  pym_src_uv_size = pym_src_uv_stride * pym_src_h / 2;

  scale = 1 << (min_cost_scale - 1);
  return kHorizonVisionSuccess;
}

int CropDetectPredictor::CropPadAndResizeRoi(hobot::vision::BBox *norm_box,
    const std::shared_ptr<hobot::vision::PymImageFrame>& pyramid,
    uint8_t **img_data, HobotXStreamImageToolsResizeInfo *resize_info,
    float *scale) {
  int ret;
  uint8_t *pym_src_y_data , *pym_src_uv_data;
  int pym_src_w, pym_src_h;
  int pym_src_y_size, pym_src_uv_size;

  int pym_src_y_stride, pym_src_uv_stride;

  GetPymScaleInfo(
      pyramid, norm_box,
      pym_src_y_data, pym_src_uv_data,
      pym_src_w, pym_src_h,
      pym_src_y_size, pym_src_uv_size,
      pym_src_y_stride, pym_src_uv_stride, *scale);

  auto scale_box = *norm_box;
  scale_box.x1 /= *scale, scale_box.x2 /= *scale;
  scale_box.y1 /= *scale, scale_box.y2 /= *scale;

  ret = CropPadAndResizeRoi(
      &scale_box,
      pym_src_y_data, pym_src_uv_data,
      pym_src_w, pym_src_h,
      pym_src_y_size, pym_src_uv_size,
      pym_src_y_stride, pym_src_uv_stride,
      img_data, resize_info);
  return ret;
}

std::vector<BaseDataPtr> CropDetectPredictor::RunSingleFrame(
    const std::vector<BaseDataPtr> &frame_input) {
  LOGD << "CropDetectPredictor RunSingleFrame";
  HOBOT_CHECK(frame_input.size() == 2);
  auto xstream_img = std::static_pointer_cast<XStreamData<
      std::shared_ptr<hobot::vision::ImageFrame>>>(frame_input[0]);
  auto pyramid = std::static_pointer_cast<hobot::vision::PymImageFrame>(
      xstream_img->value);

  auto xstream_rois = std::static_pointer_cast<BaseDataVector>(frame_input[1]);
  size_t rois_num = xstream_rois->datas_.size();

  auto frame_output_tensors = std::make_shared<BPU_TENSOR_S_LIST_PTR_LIST>();
  auto frame_input_tensors = std::make_shared<BPU_TENSOR_S_LIST_PTR_LIST>();

  frame_output_tensors->resize(rois_num);
  frame_input_tensors->resize(rois_num);

  std::vector<BaseDataPtr> frame_output;
  frame_output.emplace_back(std::make_shared<BaseDataVector>());
  RUN_PROCESS_TIME_PROFILER("Run Detect Model");
  RUN_FPS_PROFILER("Run Detect Model");
  for (size_t roi_idx = 0; roi_idx < rois_num; roi_idx++) {
    HobotXStreamImageToolsResizeInfo resize_info;
    int src_img_height = 0, src_img_width = 0;
    int padding_right = 0, padding_bottom = 0;
    float scale = 0;

    // 1. prepare data for croping
    std::shared_ptr<XStreamData<hobot::vision::BBox>> p_roi;
    p_roi = std::static_pointer_cast<XStreamData<hobot::vision::BBox>>(
        xstream_rois->datas_[roi_idx]);
    hobot::vision::BBox *norm_box = &(p_roi->value);

    if (p_roi->state_ != xstream::DataState::VALID) {
      LOGI << "BBox is not valid";
      continue;
    }

    auto async_data = std::make_shared<XStreamData<std::shared_ptr<
        hobot::vision::AsyncData>>>();
    async_data->value = std::make_shared<hobot::vision::AsyncData>();
    std::static_pointer_cast<BaseDataVector>(
        frame_output[0])->datas_.push_back(async_data);

    (*frame_input_tensors)[roi_idx] =
        std::make_shared<std::vector<BPU_TENSOR_S>>();
    (*frame_output_tensors)[roi_idx] =
        std::make_shared<std::vector<BPU_TENSOR_S>>();

    auto &input_tensors = frame_input_tensors->at(roi_idx);

    std::shared_ptr<std::vector<BPU_BBOX>> bpu_boxes =
        std::make_shared<std::vector<BPU_BBOX>>();
    bpu_boxes->emplace_back(BPU_BBOX {
        norm_box->x1, norm_box->y1, norm_box->x2, norm_box->y2,
        norm_box->score, 0, true
    });

    if (pyramid->type == "PymImageFrame") {
      uint8_t *nv12_data;
      // 2. crop and resize Roi
      CropPadAndResizeRoi(norm_box, pyramid, &nv12_data, &resize_info, &scale);
      padding_right =
          resize_info.padding_right_ / resize_info.width_ratio_;
      padding_right &= ~1;
      padding_bottom =
          resize_info.padding_bottom_ / resize_info.height_ratio_;
      padding_bottom &= ~1;

      src_img_height = resize_info.src_height_ + padding_bottom;
      src_img_width = resize_info.src_width_ + padding_right;
      {
        // 3. prepare intput tensor
        RUN_PROCESS_TIME_PROFILER("Run PrepareInputTensorData");
        RUN_FPS_PROFILER("Run PrepareInputTensorData");
        if (resize_type_ == BPU_RESIZE) {
          int img_len = src_img_width * src_img_height * 3 / 2;
          PrepareInputTensorData(
              nv12_data, img_len, src_img_width,
              src_img_height,
              input_tensors, BPU_TYPE_IMG_NV12_SEPARATE);
        } else if (resize_type_ == CPU_RESIZE) {
          int img_len = model_input_width_ * model_input_height_ * 3 / 2;
          PrepareInputTensorData(nv12_data, img_len, input_tensors);
        } else {
          HOBOT_CHECK(false) << "resize type undefined";
        }
      }
      HobotXStreamFreeImage(nv12_data);
    } else if (pyramid->type == "CVImageFrame") {
      auto cv_image = std::static_pointer_cast<
          hobot::vision::CVImageFrame>(xstream_img->value);
      // support nv12, bgr
      HOBOT_CHECK(cv_image->pixel_format ==
          HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawNV12 ||
          cv_image->pixel_format ==
              HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawBGR);
      HOBOT_CHECK(norm_box->Width() >= cv_image->Width() &&
          norm_box->Height() >= cv_image->Height());
      cv::Mat img_nv12;
      if (cv_image->pixel_format ==
          HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawBGR) {
        auto img_mat = cv_image->img;
        // bgr_to_nv12
        {
          cv::Mat yuv_mat;
          cv::cvtColor(img_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
          uint8_t *yuv = yuv_mat.ptr<uint8_t>();
          img_nv12 = cv::Mat(cv_image->Height() * 3 / 2,
                             cv_image->Width(), CV_8UC1);
          uint8_t *nv12 = img_nv12.ptr<uint8_t>();

          int uv_height = cv_image->Height() / 2;
          int uv_width = cv_image->Width() / 2;
          // copy y data
          int y_size = uv_height * uv_width * 4;
          memcpy(nv12, yuv, y_size);

          // copy uv data
          int uv_stride = uv_width * uv_height;
          uint8_t *uv_data = nv12 + y_size;
          for (int i = 0; i < uv_stride; ++i) {
            *(uv_data++) = *(yuv + y_size + i);
            *(uv_data++) = *(yuv + y_size + uv_stride + i);
          }
        }
      } else {
        img_nv12 = cv_image->img;
      }
      auto src_y_data = img_nv12.ptr<uint8_t>();
      uint8_t *nv12_data = nullptr;
      CropPadAndResizeRoi(
          norm_box,
          src_y_data, src_y_data + cv_image->Height() * cv_image->Width(),
          cv_image->Width(), cv_image->Height(),
          cv_image->Height() * cv_image->Width() ,
          cv_image->Height() * cv_image->Width() / 2,
          cv_image->Width(), cv_image->Width(),
          &nv12_data, &resize_info);
      padding_right =
          resize_info.padding_right_ / resize_info.width_ratio_;
      padding_right &= ~1;
      padding_bottom =
          resize_info.padding_bottom_ / resize_info.height_ratio_;
      padding_bottom &= ~1;

      src_img_height = resize_info.src_height_ + padding_bottom;
      src_img_width = resize_info.src_width_ + padding_right;
      {
        RUN_PROCESS_TIME_PROFILER("Run PrepareInputTensorData");
        RUN_FPS_PROFILER("Run PrepareInputTensorData");
        if (resize_type_ == BPU_RESIZE) {
          int img_len = src_img_width * src_img_height * 3 / 2;
          PrepareInputTensorData(
              nv12_data, img_len, src_img_width, src_img_height,
              input_tensors, BPU_TYPE_IMG_NV12_SEPARATE);
        } else if (resize_type_ == CPU_RESIZE) {
          int img_len = model_input_width_ * model_input_height_ * 3 / 2;
          PrepareInputTensorData(nv12_data, img_len, input_tensors);
        } else {
          HOBOT_CHECK(false) << "resize type undefined";
        }
      }
      HobotXStreamFreeImage(nv12_data);
    }
    // 4. prepare output tensor
    auto &output_tensors = frame_output_tensors->at(roi_idx);
    {
      RUN_PROCESS_TIME_PROFILER("Run PrepareOutputTensor");
      RUN_FPS_PROFILER("Run PrepareOutputTensor");
      PrepareOutputTensor(output_tensors);
    }
    // 5. run model
    BPU_RUN_CTRL_S run_ctrl_s{2};
    run_ctrl_s.resize_type = resize_type_ == BPU_RESIZE;

    std::shared_ptr<BPU_TASK_HANDLE> task_handle =
        std::make_shared<BPU_TASK_HANDLE>();
    {
      RUN_PROCESS_TIME_PROFILER("Run HB_BPU_runModel");
      RUN_FPS_PROFILER("Run HB_BPU_runModel");

      int ret = HB_BPU_runModel(bpu_model_.get(),
                                input_tensors->data(),
                                bpu_model_->input_num,
                                output_tensors->data(),
                                bpu_model_->output_num,
                                &run_ctrl_s, false, task_handle.get());
      if (ret != 0) {
        // release input&&output tensor
        ReleaseTensor(input_tensors);
        ReleaseTensor(output_tensors);
        // release task_handle
        HB_BPU_releaseTask(task_handle.get());
      }
    }

    //  in crop situation,
    //  'src_image_height' and 'src_image_width'
    //  should represent crop image added with padding value,
    //  rather than original image.
    async_data->value->src_image_height = src_img_height * scale;
    async_data->value->src_image_width = src_img_width * scale;
    async_data->value->model_input_height = model_input_height_;
    async_data->value->model_input_width = model_input_width_;

    async_data->value->input_tensors = input_tensors;
    async_data->value->output_tensors = output_tensors;
    async_data->value->task_handle = task_handle;
    async_data->value->bpu_model = bpu_model_;
    async_data->value->bpu_boxes = bpu_boxes;
  }
     // input_tensors, output_tensors传递给async_data
    return frame_output;
  }
}  // namespace xstream
