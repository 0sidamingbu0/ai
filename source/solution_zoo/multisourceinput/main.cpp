/*
 * @Description: implement of apa main
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-31 09:30:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-31 20:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include <malloc.h>
#include <signal.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "bpu_predict/bpu_predict_extension.h"
#include "hobotlog/hobotlog.hpp"
#include "json/json.h"
#include "multisourcesmartplugin/smartplugin.h"
#include "multisourcewebsocketplugin/websocketplugin.h"
#include "vioplugin/vioplugin.h"

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using std::chrono::seconds;

using horizon::vision::xproto::multisourcesmartplugin::SmartPlugin;
using horizon::vision::xproto::vioplugin::VioPlugin;
using horizon::vision::xproto::multisourcewebsocketplugin::WebsocketPlugin;

static bool exit_ = false;

static void signal_handle(int param) {
  std::cout << "recv signal " << param << ", stop" << std::endl;
  if (param == SIGINT) {
    exit_ = true;
  }
}

int main(int argc, char **argv) {
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
  if (argc < 3) {
    std::cout << "Usage: multisourceinput vio_config_file "
              << "xstream_config_file websocketplugin_config "
              << "[-i/-d/-w/-f] " << std::endl;
    return 0;
  }
  std::string vio_config_file = std::string(argv[1]);
  std::string smart_config_file = std::string(argv[2]);
  std::string websocket_config_file = std::string(argv[3]);
  std::string log_level(argv[4]);
  if (log_level == "-i") {
    SetLogLevel(HOBOT_LOG_INFO);
  } else if (log_level == "-d") {
    SetLogLevel(HOBOT_LOG_DEBUG);
  } else if (log_level == "-w") {
    std::cout << "start set.." << std::endl;
    SetLogLevel(HOBOT_LOG_WARN);
    std::cout << "after set.." << std::endl;
  } else if (log_level == "-e") {
    SetLogLevel(HOBOT_LOG_ERROR);
  } else if (log_level == "-f") {
    SetLogLevel(HOBOT_LOG_FATAL);
  } else {
    std::cout << "log option: [-i/-d/-w/-f] " << std::endl;
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
  std::ifstream ifs(websocket_config_file);
  if (!ifs.is_open()) {
    LOGF << "open websocket config file " << websocket_config_file << " failed";
    return 0;
  }
  Json::CharReaderBuilder builder;
  std::string err_json;
  Json::Value json_obj;
  bool ret = false;
  try {
    ret = Json::parseFromStream(builder, ifs, &json_obj, &err_json);
    if (!ret) {
      LOGF << "invalid config file " << websocket_config_file;
      return 0;
    }
  } catch (std::exception &e) {
    LOGF << "exception while parse config file " << websocket_config_file
         << ", " << e.what();
    return 0;
  }
  if (json_obj.isMember("display_mode")) {
    display_mode = json_obj["display_mode"].asUInt();
  } else {
    LOGF << websocket_config_file << " should set display mode";
    return 0;
  }
  if (display_mode < 0) {
    LOGF << websocket_config_file << " set display mode failed";
    return 0;
  }

  signal(SIGINT, signal_handle);
  signal(SIGPIPE, signal_handle);

  auto vio_plg = std::make_shared<VioPlugin>(vio_config_file);
  auto smart_plg = std::make_shared<SmartPlugin>(smart_config_file);
  std::shared_ptr<XPluginAsync> display_plg = nullptr;
  if (display_mode == 0) {
    LOGW << "not support this mode..";
    return 0;
  } else if (display_mode == 1) {
    LOGD << "create Web visual plugin.";
    display_plg = std::make_shared<WebsocketPlugin>(websocket_config_file);
  }
  ret = display_plg->Init();
  if (ret != 0) {
    LOGF << "display init failed";
    return -1;
  }

  ret = smart_plg->Init();
  if (ret != 0) {
    LOGF << "Failed to init smart plugin";
    return -1;
  }

  ret = vio_plg->Init();
  if (ret != 0) {
    LOGF << "Failed to init vio plugin";
    return -1;
  }

  ret = display_plg->Start();
  if (ret != 0) {
    LOGF << "display start fail";
    return -1;
  }

  ret = smart_plg->Start();
  if (ret != 0) {
    LOGF << "smart plugin start failed";
    return -1;
  }
  ret = vio_plg->Start();
  if (ret != 0) {
    LOGF << "vio plugin start failed";
    return -1;
  }

  if (run_mode == "ut") {
    std::this_thread::sleep_for(std::chrono::seconds(60));
  } else {
    while (!exit_) {
      std::this_thread::sleep_for(std::chrono::microseconds(40));
    }
  }

  vio_plg->Stop();
  vio_plg->DeInit();
  vio_plg = nullptr;
  smart_plg->Stop();
  smart_plg->DeInit();
  display_plg->Stop();
  display_plg->DeInit();
  return 0;
}
