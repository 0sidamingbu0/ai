/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: DetectPredictor.h
 * @Brief: declaration of the DetectPredictor
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-08-24 15:17:48
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-08-27 18:57:24
 */

#ifndef PREDICTMETHOD_PREDICTORS_CROPDETECTPREDICTOR_H_
#define PREDICTMETHOD_PREDICTORS_CROPDETECTPREDICTOR_H_

#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include "xstream/vision_type/include/horizon/vision_type/vision_type.hpp"
#include "xstream/imagetools/include/hobotxstream/image_tools.h"
#include "PredictMethod/Predictors/Predictors.h"
#include "hobotxsdk/xstream_data.h"
#include "bpu_predict/bpu_predict.h"
#include "bpu_predict/bpu_predict_extension.h"

namespace xstream {

class CropDetectPredictor : public Predictors {
 public:
  CropDetectPredictor() {}
  virtual ~CropDetectPredictor() {}

  std::vector<BaseDataPtr> Do(
      const std::vector<BaseDataPtr> &input,
      const xstream::InputParamPtr &param);

 private:
  std::vector<BaseDataPtr> RunSingleFrame(
      const std::vector<BaseDataPtr> &frame_input);
  int CropPadAndResizeRoi(hobot::vision::BBox *norm_box,
                          uint8_t *pym_src_y_data,
                          uint8_t *pym_src_uv_data,
                          int pym_src_w,
                          int pym_src_h,
                          int pym_src_y_size,
                          int pym_src_uv_size,
                          int pym_src_y_stride,
                          int pym_src_uv_stride,
                          uint8_t **img_data,
                          HobotXStreamImageToolsResizeInfo *resize_info);
  int CropPadAndResizeRoi(
      hobot::vision::BBox *norm_box,
      const std::shared_ptr<hobot::vision::PymImageFrame> &pyramid,
      uint8_t **img_data,
      HobotXStreamImageToolsResizeInfo *resize_info,
      float *scale);
  void CalcResizeInfo(HobotXStreamImageToolsResizeInfo *resize_info);
  int GetPymScaleInfo(
      const std::shared_ptr<hobot::vision::PymImageFrame> &pyramid,
      hobot::vision::BBox *norm_box,
      uint8_t *&pym_src_y_data,
      uint8_t *&pym_src_uv_data,
      int &pym_src_w,
      int &pym_src_h,
      int &pym_src_y_size,
      int &pym_src_uv_size,
      int &pym_src_y_stride,
      int &pym_src_uv_stride,
      float &scale);
};
}  // namespace xstream

#endif  // PREDICTMETHOD_PREDICTORS_CROPDETECTPREDICTOR_H_
