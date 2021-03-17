/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef INCLUDE_VIO_PIPE_CONFIG_H_
#define INCLUDE_VIO_PIPE_CONFIG_H_

#include <string>
#include <mutex>
#include "json/json.h"
#include "iotviomanager/vio_data_type.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

class VioPipeConfig {
 public:
     VioPipeConfig() = delete;
     VioPipeConfig(const std::string &path, const int &pipe_id) :\
       path_(path), pipe_id_(pipe_id) {}
     ~VioPipeConfig() {}
     std::string GetStringValue(const std::string &key) const;
     int GetIntValue(const std::string &key) const;
     Json::Value GetJson() const;
     bool LoadConfigFile();
     bool ParserConfig();
     bool PrintConfig();
     bool GetConfig(void *cfg);

 private:
     std::string path_;
     Json::Value json_;
     mutable std::mutex mutex_;
     int pipe_id_;
     IotVioCfg vio_cfg_ = { 0 };
};

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_VIO_PIPE_CONFIG_H_
