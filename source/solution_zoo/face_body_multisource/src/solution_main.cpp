/**
  * Copyright (c) 2020 Horizon Robotics. All rights reserved.
  * @brief     人脸/人体的多路/单路 solution demo
  * @author zhuoran.rong(zhuoran.rong@horizon.ai)
  * @date
  */

#include <signal.h>
#include <malloc.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <string>
#include <memory>
#include "hobotlog/hobotlog.hpp"
#include "vioplugin/vioplugin.h"
#include "smartplugin/smartplugin_multisource.h"
#include "websocketplugin/websocketplugin_multisource.h"
#include "bpu_predict/bpu_predict_extension.h"

using horizon::vision::xproto::multisourcesmartplugin::FaceBodySmartPlugin;
using horizon::vision::xproto::multisourcewebsocketplugin::
    FaceBodyWebsocketPlugin;
using horizon::vision::xproto::vioplugin::VioPlugin;
using std::chrono::seconds;

static volatile bool exit_ = false;

static void signal_handle(int param) {
  std::cout << "recv signal " << param << ", stop" << std::endl;
  if (param == SIGINT) {
    exit_ = true;
  }
}

int solution_main(int argc, const char **argv) {
  auto mallopt_option = getenv("MALLOC");
  auto bpu_engine_option = getenv("BPU_ENGINE");
  if (mallopt_option && !strcmp(mallopt_option, "OFF")) {
    std::cout << "turn off mallopt" << std::endl;
  } else {
    // default use mallopt
    std::cout << "turn on mallopt" << std::endl;
    mallopt(M_TRIM_THRESHOLD, 128 * 1024);
  }
  if (bpu_engine_option && !strcmp(bpu_engine_option, "group")) {
    std::cout << "use bpu group engine" << std::endl;
    HB_BPU_setGlobalConfig(BPU_GLOBAL_ENGINE_TYPE, "group");
  } else {
    // default use native engine
    std::cout << "use bpu native engine" << std::endl;
    HB_BPU_setGlobalConfig(BPU_GLOBAL_ENGINE_TYPE, "native");
  }

  std::string run_mode = "ut";
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

  if (argc == 6) {
    run_mode.assign(argv[5]);

    if (run_mode != "ut" && run_mode != "normal") {
      LOGE << "not support mode: " << run_mode;
      return 0;
    }
  }

  signal(SIGINT, signal_handle);
  signal(SIGPIPE, signal_handle);
  // signal(SIGSEGV, signal_handle);

  auto vio_plg = std::make_shared<VioPlugin>(vio_config_file);
  auto smart_plg = std::make_shared<FaceBodySmartPlugin>(smart_config_file);
  auto websocket_plg =
      std::make_shared<FaceBodyWebsocketPlugin>(websocket_config_file);

  int rv = vio_plg->Init();
  if (rv != 0) {
    LOGE << "vio plugin init failed";
    return 1;
  }
  LOGI << "vio plugin init success";

  rv = smart_plg->Init();
  if (rv < 0) {
    LOGE << "smart plugin init failed";
    return -1;
  }

  LOGI << "smart plugin init success";

  rv = websocket_plg->Init();
  if (rv < 0) {
    LOGE << "websocket plugin init failed";
    return -1;
  }

  LOGI << "websocket plugin init success";

  rv = vio_plg->Start();
  if (rv < 0) {
    LOGE << "vio plugin start failed";
    return -1;
  }
  LOGI << "vio plugin start success";

  rv = smart_plg->Start();
  if (rv < 0) {
    LOGE << "smart plugin start failed";
    return -1;
  }
  LOGI << "smart plugin start success";

  rv = websocket_plg->Start();
  if (rv < 0) {
    LOGE << "websocket plugin start failed";
    return -1;
  }
  LOGI << "websocket plugin start success";

  if (run_mode == "ut") {
    std::this_thread::sleep_for(std::chrono::seconds(60));
  } else {
    while (!exit_) {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }

  rv = vio_plg->Stop();
  if (rv < 0) {
    LOGE << "vio plugin stop failed";
    return -1;
  }
  LOGI << "vio plugin stop success";

  rv = smart_plg->Stop();
  if (rv < 0) {
    LOGE << "smart plugin stop failed";
    return -1;
  }

  rv = websocket_plg->Stop();
  if (rv < 0) {
    LOGE << "websocket plugin stop failed";
    return -1;
  }

  rv = vio_plg->DeInit();
  if (rv < 0) {
    LOGE << "vio plugin DeInit failed";
    return -1;
  }

  rv = smart_plg->DeInit();
  if (rv < 0) {
    LOGE << "smart plugin DeInit failed";
    return -1;
  }

  rv = websocket_plg->DeInit();
  if (rv < 0) {
    LOGE << "websocket plugin DeInit failed";
    return -1;
  }

  smart_plg = nullptr;
  websocket_plg = nullptr;
  LOGI << "smart plugin stop success";
  return 0;
}

