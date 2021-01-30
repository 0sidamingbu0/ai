/*!
 * Copyright (c) 2016-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     aioplugin.h
 * \Author   ronghui.zhang
 * \Version  1.0.0.0
 * \Date     2021.1.7
 * \Brief    implement of api file
 */
#ifndef INCLUDE_AIOPLUGIN_AIOPLUGIN_H_
#define INCLUDE_AIOPLUGIN_AIOPLUGIN_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include "json/json.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto_msgtype/aioplugin_data.h"
#include "alsa_device.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace aioplugin {

using horizon::vision::xproto::basic_msgtype::AioMessage;

class AioPlugin : public xproto::XPluginAsync {
 public:
  AioPlugin() = delete;
  explicit AioPlugin(std::string config_path);
  ~AioPlugin();
  int Init() override;
  int DeInit() override;
  int Start() override;
  int Stop() override;
  std::string desc() const { return "AioPlugin"; }

 private:
  int MicphoneGetThread();
  int ParseConfig(std::string config_file);
  std::shared_ptr<std::thread> micphone_thread_;
  alsa_device_t *micphone_device_;
  bool exit_;
  int audio_num_;
  std::string config_path_;
  int micphone_rate_;
  int micphone_chn_;
  int micphone_buffer_time_;
  int micphone_nperiods_;
  int micphone_period_size_;
};

}  // namespace aioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif
