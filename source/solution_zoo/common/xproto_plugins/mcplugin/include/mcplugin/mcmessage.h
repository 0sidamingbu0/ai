/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Fei cheng
 * @Mail: fei.cheng@horizon.ai
 * @Date: 2019-09-14 20:38:52
 * @Version: v0.0.1
 * @Brief: mcplugin data on xpp.
 * @Last Modified by: Fei cheng
 * @Last Modified time: 2019-09-14 22:41:30
 */
#ifndef APP_INCLUDE_PLUGIN_MCPLUGIN_MCMESSAGE_H_
#define APP_INCLUDE_PLUGIN_MCPLUGIN_MCMESSAGE_H_
#include <string>
#include <vector>
#include <memory>

#include "common_data.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace mcplugin {

using horizon::vision::xproto::XProtoMessagePtr;
using horizon::vision::xproto::XProtoMessage;

#define TYPE_MC_UPSTREAM_MESSAGE "XPLUGIN_MC_UPSTREAM_MESSAGE"
#define TYPE_MC_DOWNSTREAM_MESSAGE "XPLUGIN_MC_DOWMSTREAM_MESSAGE"

class MCMessage : public XProtoMessage {
 public:
  MCMessage() = delete;
  explicit MCMessage(ConfigMessageType type,
                     ConfigMessageMask mask,
                     uint64_t msg_id = 0) {
    type_ = TYPE_MC_DOWNSTREAM_MESSAGE;
    message_mask_ = mask;
    message_type_ = type;
    msg_id_ = msg_id;
  }
  explicit MCMessage(ConfigMessageType type, ConfigMessageMask mask,
                     std::shared_ptr<BaseConfigData> data,
                     std::string direction, uint64_t msg_id = 0) {
    if (direction == "up") {
      type_ = TYPE_MC_UPSTREAM_MESSAGE;
    } else {
      type_ = TYPE_MC_DOWNSTREAM_MESSAGE;
    }

    message_mask_ = mask;
    message_type_ = type;
    message_data_ = data;
    msg_id_ = msg_id;
  }
  ~MCMessage() = default;

  bool IsMessageValid(ConfigMessageMask mask);
  ConfigMessageType GetMessageType() { return message_type_; }
  std::shared_ptr<BaseConfigData> GetMessageData() { return message_data_; }
  std::string Serialize() override;

 private:
  ConfigMessageType message_type_;
  ConfigMessageMask message_mask_;
  std::shared_ptr<BaseConfigData> message_data_;
  uint64_t msg_id_;
};

}  // namespace mcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // APP_INCLUDE_PLUGIN_MCPLUGIN_MCMESSAGE_H_
