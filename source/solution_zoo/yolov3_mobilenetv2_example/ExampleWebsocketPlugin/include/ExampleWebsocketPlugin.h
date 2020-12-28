/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file ExampleWebsocketplugin.h
 * @brief
 * @author ronghui.zhang
 * @email ronghui.zhang@horizon.ai
 *
 *
 * */
#ifndef INCLUDE_EXAMPLEWEBSOCKETPLUGIN_EXAMPLEWEBSOCKETPLUGIN_H_
#define INCLUDE_EXAMPLEWEBSOCKETPLUGIN_EXAMPLEWEBSOCKETPLUGIN_H_

#include <string>
#include "websocketplugin/websocketplugin.h"
#include "../../ExampleSmartPlugin/include/ExampleSmartPlugin.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace ExampleWebsocketPlugin {

using horizon::vision::xproto::websocketplugin::WebsocketPlugin;

class ExampleWebsocketPlugin : public WebsocketPlugin {
 public:
  explicit ExampleWebsocketPlugin(std::string config_path);

 protected:
  std::string GetSmartMessageType() {
    return EXAMPLE_SMART_MESSAGE;
  }
};
}  // namespace ExampleWebsocketPlugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif
