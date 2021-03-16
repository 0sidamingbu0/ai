/*!
 * Copyright (c) 2016-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     uac_server.h
 * \Author   ronghui.zhang
 * \Version  1.0.0.0
 * \Date     2021/1/7
 * \Brief    implement of api header
 */

#ifndef INCLUDE_UVCPLUGIN_UAC_SERVER_H_
#define INCLUDE_UVCPLUGIN_UAC_SERVER_H_
#include <string>
#include <thread>
#include <mutex>
#include <fstream>
#include "json/json.h"
#include "alsa_device.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace Uvcplugin {

class UacServer {
 public:
  UacServer() = delete;
  explicit UacServer(std::string config_path);
  ~UacServer() {
    DeInit();
  }
  int Init();
  int Start();
  int DeInit();
  int Stop();
  int SendUacData(char* buffer, int size);

 private:
  int ParseConfig(std::string config_path);

 private:
  alsa_device_t* uac_device_;
  std::string config_path_;
  int uac_rate_;
  int uac_chn_;
  int uac_buffer_time_;
  int uac_nperiods_;
};
}  // namespace Uvcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif
