/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include "multisourcesmartplugin/util.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"
#include "horizon/vision_type/vision_type_util.h"
#include "hobotlog/hobotlog.hpp"
#include "opencv2/opencv.hpp"

#include "hobotxsdk/xstream_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {

static void check_pyramid(hobot::vision::PymImageFrame *img_info) {
  for (auto i = 0; i < 6; ++i) {
    auto index = i * 4;
    HOBOT_CHECK(img_info->down_scale[index].height);
    HOBOT_CHECK(img_info->down_scale[index].stride);
  }
}

XStreamImageFramePtr *ImageConversion(const HorizonVisionImage &c_img) {
  auto xstream_img = new XStreamImageFramePtr();
  xstream_img->type_ = "ImageFrame";
  auto image_type = c_img.pixel_format;
  switch (image_type) {
    case kHorizonVisionPixelFormatNone: {
      LOGI << "kHorizonVisionPixelFormatNone, data size is " << c_img.data_size;
      auto cv_img = std::make_shared<hobot::vision::CVImageFrame>();
      std::vector<unsigned char>
          buf(c_img.data, c_img.data + c_img.data_size);
      cv_img->img =
          cv::imdecode(buf, cv::IMREAD_COLOR);
      HOBOT_CHECK(!cv_img->img.empty())
      << "Invalid image , failed to create cvmat for"
         " kHorizonVisionPixelFormatNone type image";
      cv_img->pixel_format =
          HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawBGR;
      xstream_img->value = cv_img;
      break;
    }
    case kHorizonVisionPixelFormatRawBGR: {
      auto cv_img = std::make_shared<hobot::vision::CVImageFrame>();
      cv_img->img =
          cv::Mat(c_img.height, c_img.width, CV_8UC3, c_img.data);
      HOBOT_CHECK(!cv_img->img.empty())
      << "Invalid image , failed to create cvmat for"
         " kHorizonVisionPixelFormatRawBGR type image";
      cv_img->pixel_format =
          HorizonVisionPixelFormat::kHorizonVisionPixelFormatRawBGR;
      xstream_img->value = cv_img;
      break;
    }
    case kHorizonVisionPixelFormatPYM: {
      auto pym_img = std::make_shared<hobot::vision::PymImageFrame>();
      hobot::vision::PymImageFrame *img_info =
        reinterpret_cast<hobot::vision::PymImageFrame *>(c_img.data);
      HOBOT_CHECK(img_info);
      check_pyramid(img_info);
      *pym_img = *img_info;  // default assign oper
      xstream_img->value = pym_img;
      break;
    }
    default: {
      LOGF << "No support image type " << image_type;
    }
  }
  return xstream_img;
}

HorizonVisionImage *ImageConversion(uint8_t *data, int data_size, int width,
                                    int height, int stride, int stride_uv,
                                    HorizonVisionPixelFormat format) {
  HOBOT_CHECK(data) << "ImageConversion null input";
  HorizonVisionImage *c_img;
  HorizonVisionAllocImage(&c_img);
  c_img->pixel_format = format;

  c_img->data = static_cast<uint8_t *>(std::calloc(data_size, sizeof(uint8_t)));

  memcpy(c_img->data, data, data_size);
  c_img->data_size = data_size;
  c_img->width = width;
  c_img->height = height;
  c_img->stride = stride;
  c_img->stride_uv = stride_uv;
  return c_img;
}

HorizonVisionImage *ImageConversion(const ImageFramePtr &cpp_img) {
  HOBOT_CHECK(cpp_img) << "ImageConversion null input";
  HorizonVisionImage *c_img;
  HorizonVisionAllocImage(&c_img);
  c_img->pixel_format = cpp_img->pixel_format;

  c_img->data =
      static_cast<uint8_t *>(std::calloc(cpp_img->DataSize(), sizeof(uint8_t)));

  memcpy(c_img->data, reinterpret_cast<void *>(cpp_img->Data()),
         cpp_img->DataSize());
  c_img->data_size = cpp_img->DataSize();
  c_img->width = cpp_img->Width();
  c_img->height = cpp_img->Height();
  c_img->stride = cpp_img->Stride();
  c_img->stride_uv = cpp_img->StrideUV();
  return c_img;
}

XStreamImageFramePtr *ImageFrameConversion(
    const HorizonVisionImageFrame *c_img_frame) {
  auto xstream_img_frame = ImageConversion(c_img_frame->image);
  xstream_img_frame->value->time_stamp = c_img_frame->time_stamp;
  xstream_img_frame->value->channel_id = c_img_frame->channel_id;
  xstream_img_frame->value->frame_id = c_img_frame->frame_id;
  return xstream_img_frame;
}

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon


