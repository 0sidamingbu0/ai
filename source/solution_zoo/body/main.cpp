/**
 * Copyright (c) 2019 Horizon Robotics. All rights reserved.
 * @file main.cpp
 * @brief
 * @author fei.cheng
 * @email fei.cheng@horizon.ai
 *
 *
 * */

#include <signal.h>
#include <unistd.h>
#include <malloc.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "json/json.h"
#include "hobotlog/hobotlog.hpp"
#include "smartplugin/smartplugin.h"
#include "vioplugin/vioplugin.h"
#include "visualplugin/visualplugin.h"
#include "websocketplugin/websocketplugin.h"
#include "bpu_predict/bpu_predict_extension.h"
#ifdef X3
#include "uvcplugin/uvcplugin.h"
#include "aioplugin/aioplugin.h"
#endif

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using std::chrono::seconds;

// using horizon::vision::xproto::hbipcplugin::HbipcPlugin;
using horizon::vision::xproto::smartplugin::SmartPlugin;
using horizon::vision::xproto::vioplugin::VioPlugin;
using horizon::vision::xproto::visualplugin::VisualPlugin;
using horizon::vision::xproto::websocketplugin::WebsocketPlugin;
using horizon::vision::xproto::XPluginAsync;
#ifdef X3
using horizon::vision::xproto::Uvcplugin::UvcPlugin;
using horizon::vision::xproto::aioplugin::AioPlugin;
#endif
static bool exit_ = false;

static void signal_handle(int param) {
  std::cout << "recv signal " << param << ", stop" << std::endl;
  if (param == SIGINT) {
    exit_ = true;
  }
}

int main(int argc, char **argv) {
  HB_BPU_setGlobalConfig(BPU_GLOBAL_CONFIG_MAX_TASK_NUM, "128");
  auto mallopt_option = getenv("MALLOC");
  auto bpu_engine_option = getenv("BPU_ENGINE");
  if (mallopt_option && !strcmp(mallopt_option, "OFF")) {
    std::cout << "turn off mallopt" << std::endl;
  } else {
    // default use mallopt
    std::cout << "turn on mallopt" << std::endl;
    mallopt(M_TRIM_THRESHOLD, 128 * 1024);
  }
  if (bpu_engine_option && !strcmp(bpu_engine_option, "native")) {
    std::cout << "use bpu native engine" << std::endl;
    HB_BPU_setGlobalConfig(BPU_GLOBAL_ENGINE_TYPE, "native");
  } else {
    // default use group engine
    std::cout << "use bpu group engine" << std::endl;
    HB_BPU_setGlobalConfig(BPU_GLOBAL_ENGINE_TYPE, "group");
  }

  std::string run_mode = "ut";

  if (argc < 5) {
    std::cout << "Usage: smart_main vio_config_file "
              << "xstream_config_file visualplugin_config "
              << "[-i/-d/-w/-f] " << std::endl;
    return 0;
  }

  std::string vio_config_file = std::string(argv[1]);
  std::string smart_config_file = std::string(argv[2]);
  std::string visual_config_file = std::string(argv[3]);

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

  // parse output display mode config
  int display_mode = -1;
#ifdef X3
  int audio_enable = 0;
#endif
  std::ifstream ifs(visual_config_file);
  if (!ifs.is_open()) {
    LOGF << "open config file " << visual_config_file << " failed";
    return 0;
  }
  Json::CharReaderBuilder builder;
  std::string err_json;
  Json::Value json_obj;
  try {
    bool ret = Json::parseFromStream(builder, ifs, &json_obj, &err_json);
    if (!ret) {
      LOGF << "invalid config file " << visual_config_file;
      return 0;
    }
  } catch (std::exception &e) {
    LOGF << "exception while parse config file " << visual_config_file << ", "
         << e.what();
    return 0;
  }
  if (json_obj.isMember("display_mode")) {
    display_mode = json_obj["display_mode"].asUInt();
  } else {
    LOGF << visual_config_file << " should set display mode";
    return 0;
  }
  if (display_mode < 0) {
    LOGF << visual_config_file << " set display mode failed";
    return 0;
  }
#ifdef X3
  if (json_obj.isMember("audio_enable")) {
    audio_enable = json_obj["audio_enable"].asInt();
  }
#endif

  signal(SIGINT, signal_handle);
  signal(SIGPIPE, signal_handle);

#ifdef X3
  std::shared_ptr<AioPlugin> aio_plg = nullptr;
  if (audio_enable) {
    aio_plg = std::make_shared<AioPlugin>(visual_config_file);
  }
#endif
  auto vio_plg = std::make_shared<VioPlugin>(vio_config_file);
  auto smart_plg = std::make_shared<SmartPlugin>(smart_config_file);
  std::shared_ptr<XPluginAsync> output_plg = nullptr;
  // create output plugin: QtVisualPlugin/WebsocketPlugin/UvcPlugin

  if (0 == display_mode) {
    LOGI << "create QT Visual Plugin";
    output_plg = std::make_shared<VisualPlugin>(visual_config_file);
  } else if (1 == display_mode) {
    LOGI << "create WebSocket plugin";
    output_plg = std::make_shared<WebsocketPlugin>(visual_config_file);
  }
#ifdef X3
  if (2 == display_mode) {
    LOGI << "create UVC plugin";
    output_plg = std::make_shared<UvcPlugin>(visual_config_file);
  }
#endif

  auto ret = vio_plg->Init();
  if (ret != 0) {
    LOGE << "Failed to init vio";
    return 1;
  }

#ifdef X3
  if (audio_enable) {
    ret = aio_plg->Init();
    if (ret != 0) {
      LOGE << "Failed to init aio";
      return 1;
    }
  }
#endif

  ret = smart_plg->Init();
  if (ret != 0) {
    LOGE << "Failed to init smart plugin";
    return 2;
  }
  if (output_plg) {
    ret = output_plg->Init();
    if (ret != 0) {
      LOGE << "output plugin init failed";
      return 3;
    }
    ret = output_plg->Start();
    if (ret != 0) {
      LOGE << "output plugin start failed";
      return 3;
    }
  }

  vio_plg->Start();

#ifdef X3
  if (audio_enable) {
    aio_plg->Start();
  }
#endif
  smart_plg->Start();

  if (run_mode == "ut") {
    std::this_thread::sleep_for(std::chrono::seconds(60));
  } else {
    while (!exit_) {
      std::this_thread::sleep_for(std::chrono::microseconds(40));
    }
  }
  if (output_plg) {
    output_plg->Stop();
  }
  output_plg->DeInit();
  smart_plg->Stop();
  smart_plg->DeInit();

#ifdef X3
  if (audio_enable) {
    aio_plg->Stop();
    aio_plg->DeInit();
    aio_plg = nullptr;
  }
#endif

  vio_plg->Stop();
  vio_plg->DeInit();
  vio_plg = nullptr;
  return 0;
}
