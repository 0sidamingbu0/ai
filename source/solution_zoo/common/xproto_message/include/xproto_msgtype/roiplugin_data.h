/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: xue.liang
 * @Mail: xue.liang@horizon.ai
 * @Date: 2020-12-15 20:38:52
 * @Version: v0.0.1
 * @Brief: roiplugin impl based on xpp.
 * @Last Modified by: xue.liang
 * @Last Modified time: 2020-12-16 22:41:30
 */

#ifndef XPROTO_MSGTYPE_ROIPLUGIN_DATA_H_
#define XPROTO_MSGTYPE_ROIPLUGIN_DATA_H_

#include <memory>
#include <string>

#include "xproto/message/pluginflow/flowmsg.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace basic_msgtype {

#define TYPE_ROI_SMART_MESSAGE "XPLUGIN_ROI_SMART_MESSAGE"

struct RoiSmartMessage : XProtoMessage {
  int channel_id;
  int frame_fps;
  uint64_t time_stamp;
  uint64_t frame_id;
  std::string image_name;
  RoiSmartMessage() { type_ = TYPE_ROI_SMART_MESSAGE; }
  virtual ~RoiSmartMessage() = default;

  std::string Serialize() override { return "Default roi smart message"; };
  virtual std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) {
    return "";
  }
};
using RoiSmartMessagePtr = std::shared_ptr<RoiSmartMessage>;
}  // namespace basic_msgtype
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // XPROTO_MSGTYPE_ROIPLUGIN_DATA_H_
