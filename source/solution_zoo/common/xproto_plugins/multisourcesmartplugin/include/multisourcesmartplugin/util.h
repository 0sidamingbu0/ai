/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCESMARTPLUGIN_VISION_UTIL_H_
#define INCLUDE_MULTISOURCESMARTPLUGIN_VISION_UTIL_H_
#include <cstdint>
#include <memory>

#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"
#include "hobotxsdk/xstream_sdk.h"

using hobot::vision::ImageFrame;
using hobot::vision::BBox;

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {

using ImageFramePtr = std::shared_ptr<hobot::vision::ImageFrame>;
using XStreamImageFramePtr = xstream::XStreamData<ImageFramePtr>;
using XStreamBaseDataVectorPtr = std::shared_ptr<xstream::BaseDataVector>;

HorizonVisionImage *ImageConversion(uint8_t *data,
                                    int data_size,
                                    int width,
                                    int height,
                                    int stride,
                                    int stride_uv,
                                    HorizonVisionPixelFormat format);

HorizonVisionImage *ImageConversion(const ImageFramePtr &cpp_img);
XStreamImageFramePtr *ImageConversion(const HorizonVisionImage &c_img);
XStreamImageFramePtr *ImageFrameConversion(const HorizonVisionImageFrame *);

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  //  INCLUDE_MULTISOURCESMARTPLUGIN_VISION_UTIL_H_
