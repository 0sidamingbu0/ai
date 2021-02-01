/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file websocketplugin_multisource.cpp
 * @brief
 * @author xudong.du
 * @email xudong.du@horizon.ai
 *
 *
 * */
#include "websocketplugin/websocketplugin_multisource.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {

FaceBodyWebsocketPlugin::FaceBodyWebsocketPlugin(std::string config_file)
    : WebsocketPlugin(config_file) {}
}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
