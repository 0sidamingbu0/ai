/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: xue.liang
 * @Mail: xue.liang@horizon.ai
 * @Date: 2020-12-15 20:38:52
 * @Version: v0.0.1
 * @Brief:
 * @Last Modified by: xue.liang
 * @Last Modified time: 2020-12-16 22:41:30
 */

#ifndef INCLUDE_SMARTPLUGIN_TCLCONVERTOR_H_
#define INCLUDE_SMARTPLUGIN_TCLCONVERTOR_H_

#include <future>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "hobotxsdk/xstream_sdk.h"
#include "horizon/vision/util.h"
#include "horizon/vision_type/vision_error.h"
#include "horizon/vision_type/vision_type.hpp"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto_msgtype/smartplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin {
using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;

class KeyPointConvertor {
 public:
  KeyPointConvertor() = default;
  ~KeyPointConvertor() = default;

 public:
  static void ConverKeyPoint(xstream::OutputDataPtr xstream_out);

 private:
  static void ConvertFaceLmk(xstream::BaseDataVector *face_lmks);
  static void ConvertKps(xstream::BaseDataVector *kps_list,
                             xstream::BaseDataVector *body_box_list,
                             xstream::BaseDataVector *head_box_list,
                             xstream::BaseDataVector *face_lmks,
                             xstream::BaseDataVector *face_box_list);
};

}  // namespace smartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_SMARTPLUGIN_TCLCONVERTOR_H_
