/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Songshan Gong
 * @Mail: songshan.gong@horizon.ai
 * @Date: 2019-08-02 03:42:16
 * @Version: v0.0.1
 * @Brief:  convert to xstream inputdata from input VioMessage
 * @Note:  extracted from xperson repo.
 * @Last Modified by: Songshan Gong
 * @Last Modified time: 2019-09-29 02:57:24
 */
#ifndef INCLUDE_ROIPLUGIN_CONVERT_H_
#define INCLUDE_ROIPLUGIN_CONVERT_H_

#include <string>
#include <vector>

#include "hobotxsdk/xstream_data.h"
#include "hobotxsdk/xstream_sdk.h"
#include "horizon/vision_type/vision_msg.h"
#include "horizon/vision_type/vision_msg.hpp"
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"
#include "horizon/vision_type/vision_type_util.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace roiplugin {
class Convertor {
 public:
  static HorizonVisionSmartFrame *ConvertOutputToSmartFrame(
      const xstream::OutputDataPtr &xroc_output);

  static void ConvertOutputToSmartFrame(
      const xstream::OutputDataPtr &xroc_output,
      HorizonVisionSmartFrame *smart_frame);

 public:
  static int image_compress_quality;
};

}  // namespace roiplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  //  INCLUDE_ROIPLUGIN_CONVERT_H_
