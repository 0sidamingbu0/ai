/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file ExampleWebsocketplugin.cpp
 * @brief
 * @author ronghui.zhang
 * @email ronghui.zhang@horizon.ai
 *
 *
 * */
#include "ExampleWebsocketPlugin.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace ExampleWebsocketPlugin {

ExampleWebsocketPlugin::ExampleWebsocketPlugin(std::string config_file)
: WebsocketPlugin(config_file) {
}
}  // namespace ExampleWebsocketPlugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
