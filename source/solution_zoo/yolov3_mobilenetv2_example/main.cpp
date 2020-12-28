/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file main.cpp
 * @brief
 * @author hangjun.yang
 * @email hangjun.yang@horizon.ai
 *
 *
 * */

#include <iostream>
#include "json/json.h"
#include "hobotlog/hobotlog.hpp"
#include "vioplugin/vioplugin.h"
#include "ExampleSmartPlugin.h"
#include "ExampleWebsocketPlugin.h"

using horizon::vision::xproto::vioplugin::VioPlugin;
using horizon::vision::xproto::ExampleSmartPlugin::ExampleSmartPlugin;
using horizon::vision::xproto::ExampleWebsocketPlugin::ExampleWebsocketPlugin;

static bool exit_ = false;
static void signal_handle(int param) {
  std::cout << "recv signal " << param << ", stop" << std::endl;
  if (param == SIGINT) {
    exit_ = true;
  }
}

int main(int argc, char **argv) {
  std::string vio_config_file = std::string(argv[1]);
  std::string smart_config_file = std::string(argv[2]);
  std::string websocket_config_file = std::string(argv[3]);

  std::string log_level(argv[4]);
  if (log_level == "-i") {
    SetLogLevel(HOBOT_LOG_INFO);
  } else if (log_level == "-d") {
    SetLogLevel(HOBOT_LOG_DEBUG);
  } else if (log_level == "-w") {
    SetLogLevel(HOBOT_LOG_WARN);
  } else if (log_level == "-e") {
    SetLogLevel(HOBOT_LOG_ERROR);
  } else if (log_level == "-f") {
    SetLogLevel(HOBOT_LOG_FATAL);
  } else {
    LOGE << "log option: [-i/-d/-w/-f] ";
    return 0;
  }

  signal(SIGINT, signal_handle);
  signal(SIGPIPE, signal_handle);

  // 创建plugin对象
  auto vio_plg = std::make_shared<VioPlugin>(vio_config_file);
  auto smart_plg = std::make_shared<ExampleSmartPlugin>(smart_config_file);
  auto websocket_plg =
  std::make_shared<ExampleWebsocketPlugin>(websocket_config_file);

  auto ret = vio_plg->Init();
  if (ret != 0) {
    return -1;
  }

  ret = smart_plg->Init();
  if (ret != 0) {
    return -1;
  }

  ret = websocket_plg->Init();
  if (ret != 0) {
    return -1;
  }

  vio_plg->Start();
  smart_plg->Start();
  websocket_plg->Start();

  while (!exit_) {
    std::this_thread::sleep_for(std::chrono::microseconds(40));
  }

  vio_plg->Stop();
  vio_plg->DeInit();
  vio_plg = nullptr;
  smart_plg->Stop();
  websocket_plg->Stop();
  smart_plg->DeInit();
  websocket_plg->DeInit();
  return 0;
}
