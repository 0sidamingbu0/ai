/*!
 * Copyright (c) 2016-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     uvc_server.cpp
 * \Author   ronghui.zhang
 * \Version  1.0.0.0
 * \Date     2021.1.7
 * \Brief    implement of api file
 */
#include "uac_server.h"

#include "alsa_device.h"
#include "hobotlog/hobotlog.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace Uvcplugin {

UacServer::UacServer(std::string config_path) { config_path_ = config_path; }

int UacServer::ParseConfig(std::string config_file) {
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    LOGF << "open config file " << config_file << " failed";
    return -1;
  }
  Json::CharReaderBuilder builder;
  std::string err_json;
  Json::Value json_obj;
  try {
    bool ret = Json::parseFromStream(builder, ifs, &json_obj, &err_json);
    if (!ret) {
      LOGF << "invalid config file " << config_file;
      return -1;
    }
  } catch (std::exception &e) {
    LOGF << "exception while parse config file " << config_file << ", "
         << e.what();
    return -1;
  }

  if (json_obj.isMember("uac_rate")) {
    uac_rate_ = json_obj["uac_rate"].asInt();
  }

  if (json_obj.isMember("uac_chn")) {
    uac_chn_ = json_obj["uac_chn"].asInt();
  }

  if (json_obj.isMember("uac_buffer_time")) {
    uac_buffer_time_ = json_obj["uac_buffer_time"].asInt();
  }

  if (json_obj.isMember("uac_nperiods")) {
    uac_nperiods_ = json_obj["uac_nperiods"].asInt();
  }

  return 0;
}

int UacServer::Init() {
  int ret = -1;

  ret = ParseConfig(config_path_);
  if (ret != 0) {
    LOGE << "UacServer::Init failed";
    return -1;
  }
  uac_device_ = alsa_device_allocate();

  if (!uac_device_) {
    return -1;
  }
  /* init uac recorder device*/
  uac_device_->name = const_cast<char *>("uac_playback");
  uac_device_->format = SND_PCM_FORMAT_S16;
  uac_device_->direct = SND_PCM_STREAM_PLAYBACK;
  uac_device_->rate = uac_rate_;
  uac_device_->channels = uac_chn_;
  uac_device_->buffer_time = uac_buffer_time_;  // use default buffer time
  uac_device_->nperiods = uac_nperiods_;

  ret = alsa_device_init(uac_device_);
  if (ret < 0) {
    LOGE << "alsa_device_init failed";
    return -1;
  }

  return 0;
}

int UacServer::Start() { return 0; }

int UacServer::Stop() { return 0; }

int UacServer::DeInit() {
  if (uac_device_) {
    alsa_device_deinit(uac_device_);
    alsa_device_free(uac_device_);
  }

  return 0;
}

int UacServer::SendUacData(char *buffer, int size) {
  int ret = -1;
  auto start_time = std::chrono::system_clock::now();
  ret = alsa_device_write(uac_device_, buffer, size);
  if (ret < 0) {
    LOGE << "send uac data failed";
    return -1;
  }
  auto end_time = std::chrono::system_clock::now();
  auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  LOGV << "SendUacData succeed cost " << cost_time.count();

  return 0;
}

}  // namespace Uvcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
