/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file ExampleWebsocketplugin.h
 * @brief
 * @author ronghui.zhang
 * @email ronghui.zhang@horizon.ai
 *
 *
 * */
#ifndef INCLUDE_MULTISOURCE_WEBSOCKETPLUGIN_H_
#define INCLUDE_MULTISOURCE_WEBSOCKETPLUGIN_H_

#include <string>

#include "../smartplugin/smartplugin_multisource.h"
#include "multisourcewebsocketplugin/websocketplugin.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {

using horizon::vision::xproto::multisourcewebsocketplugin::WebsocketPlugin;

class FaceBodyWebsocketPlugin : public WebsocketPlugin {
 public:
  explicit FaceBodyWebsocketPlugin(std::string config_path);

 protected:
  std::string GetSmartMessageType() { return FACE_BODY_SMART_MESSAGE; }
};
}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif
