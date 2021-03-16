/*
 * @Description: sample
 * @Author: songhan.gong@horizon.ai
 * @Date: 2019-09-24 15:33:49
 * @LastEditors: hao.tian@horizon.ai
 * @LastEditTime: 2019-10-14 21:06:18
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#include <signal.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "hobotlog/hobotlog.hpp"
#include "vioplugin/vioplugin.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto/plugin/xpluginasync.h"

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;

using horizon::vision::xproto::basic_msgtype::VioMessage;
using horizon::vision::xproto::vioplugin::VioPlugin;

using std::chrono::milliseconds;

#define TYPE_IMAGE_MESSAGE "XPLUGIN_IMAGE_MESSAGE"
#define TYPE_DROP_MESSAGE "XPLUGIN_DROP_MESSAGE"

// 视频帧消费插件
class VioConsumerPlugin : public XPluginAsync {
 public:
  VioConsumerPlugin() = default;
  ~VioConsumerPlugin() = default;

  // 初始化,订阅消息
  int Init() {
    // 订阅视频帧消息
    RegisterMsg(TYPE_IMAGE_MESSAGE, std::bind(&VioConsumerPlugin::OnGetImage,
                                              this, std::placeholders::_1));
    return XPluginAsync::Init();
  }

  int OnGetImage(XProtoMessagePtr msg) {
    // feed video frame to xstreamsdk.
    // 1. parse valid frame from msg
    auto valid_frame = std::static_pointer_cast<VioMessage>(msg);
    valid_frame.get();

    return 0;
  }

  std::string desc() const { return "VioConsumerPlugin"; }

  // 启动plugin
  int Start() {
    std::cout << "plugin start" << std::endl;
    return 0;
  }

  // 停止plugin
  int Stop() {
    std::cout << "plugin stop" << std::endl;
    return 0;
  }
};

struct SmartContext {
  volatile bool exit;
  SmartContext() : exit(false) {}
};

SmartContext g_ctx;

static void signal_handle(int param) {
  std::cout << "recv signal " << param << ", stop" << std::endl;
  if (param == SIGINT) {
    g_ctx.exit = true;
  }
}

int main(int argc, char **argv) {
  std::string vio_config_file;
  std::string log_level;
  int loop_mode = 0;
  if (argc < 3) {
    std::cout << "Usage: ./sample vio_config_file loop_mode"
              << " [-i/-d/-w/-f] " << std::endl;
  }
  if (argv[1] == nullptr) {
    std::cout << "set default vio config:"
              << " [./configs/vio_config.json.x3dev.fb]" << std::endl;
    vio_config_file = "./configs/vio_config.json.x3dev.fb";
  } else {
    vio_config_file = argv[1];
  }
  if (argv[2] != nullptr) {
    loop_mode = std::stoi(argv[2]);
  }
  if (argv[3] == nullptr) {
    std::cout << "set default log level: [-i] ";
    log_level = "-i";
  } else {
    log_level = argv[3];
  }
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
    SetLogLevel(HOBOT_LOG_INFO);
    LOGW << "set default log level: [-i] ";
  }

  signal(SIGINT, signal_handle);
  signal(SIGPIPE, signal_handle);
  signal(SIGSEGV, signal_handle);

  auto vio_plg = std::make_shared<VioPlugin>(vio_config_file);

  vio_plg->Init();

  if (loop_mode == 0) {
    vio_plg->Start();
    while (!g_ctx.exit) {
      std::this_thread::sleep_for(milliseconds(40));
    }
  } else {
    std::this_thread::sleep_for(std::chrono::microseconds(1000 * 1000 * 5));
    while (!g_ctx.exit) {
      LOGW << "cam start\n\n";
      vio_plg->CamStart();
      std::this_thread::sleep_for(std::chrono::microseconds(1000 * 1000 * 5));
      LOGW << "cam stop\n\n";
      vio_plg->CamStop();
      LOGW << "fb start\n\n";
      vio_plg->FbStart();
      std::this_thread::sleep_for(std::chrono::microseconds(1000 * 1000 * 5));
      LOGW << "fb stop\n\n";
      vio_plg->FbStop();
    }
  }

  LOGI << "vio sample quit\n\n";
  if (loop_mode == 0) {
    vio_plg->Stop();
  }
  vio_plg->DeInit();

  return 0;
}
