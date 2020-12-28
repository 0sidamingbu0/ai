/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Fei cheng
 * @Mail: fei.cheng@horizon.ai
 * @Date: 2019-09-14 20:38:52
 * @Version: v0.0.1
 * @Brief: mcplugin impl based on xpp.
 * @Last Modified by: Fei cheng
 * @Last Modified time: 2019-09-14 22:41:30
 */

#ifndef APP_INCLUDE_PLUGIN_MCPLUGIN_MCPLUGIN_H_
#define APP_INCLUDE_PLUGIN_MCPLUGIN_MCPLUGIN_H_

#include <future>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xproto/msgtype/include/xproto_msgtype/smartplugin_data.h"
#include "xproto/msgtype/include/xproto_msgtype/vioplugin_data.h"
#include "mcplugin/mcmessage.h"
#include "hobotlog/hobotlog.hpp"
#include "utils/jsonConfigWrapper.h"
#include "xproto/utils/singleton.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
#include "xproto_msgtype/uvcplugin_data.h"
#include "xproto/plugin/xpluginasync.h"
#include "utils/votmodule.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace mcplugin {
using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using horizon::vision::xproto::basic_msgtype::UvcMessage;
using horizon::vision::xproto::basic_msgtype::TransportMessage;
using horizon::vision::xproto::basic_msgtype::SmartMessage;
using horizon::vision::xproto::basic_msgtype::VioMessage;
using horizon::vision::xproto::basic_msgtype::APImageMessage;
using MCMsgPtr = std::shared_ptr<MCMessage>;

class PluginContext : public hobot::CSingleton<PluginContext> {
 public:
  PluginContext() : exit(false) { plugins.clear(); }
  ~PluginContext() = default;

 public:
  bool exit;
  uint32_t basic_plugin_cnt = 3;
  std::vector<std::shared_ptr<XPluginAsync>> plugins;
};

class MCPlugin : public XPluginAsync {
 public:
  MCPlugin() = default;
  explicit MCPlugin(const std::string& config_file);

  ~MCPlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;

 private:
  int StartPlugin(uint64 msg_id);
  int StopPlugin(uint64 msg_id);
  int OnGetUvcResult(const XProtoMessagePtr& msg);
  int OnGetSmarterResult(const XProtoMessagePtr& msg);
  int OnGetVioResult(const XProtoMessagePtr& msg);
  void ParseConfig();
  int GetCPStatus();
  int GetCPLogLevel();

 private:
  enum cp_status {
    CP_READY,
    CP_STARTING,
    CP_WORKING,
    CP_STOPPING,
    CP_ABNORMAL,
    CP_UPDATING,
  };
  std::atomic<bool> is_running_;
  std::string config_file_;
  std::shared_ptr<JsonConfigWrapper> config_;
  cp_status cp_status_;
  int log_level_;

  void ConstructMsgForCmd(const x3::InfoMessage& InfoMsg, uint64 msg_id = 0);
  // std::string profile_log_file_;
  std::map<ConfigMessageType, std::function<int(uint64)>> cmd_func_map = {
          {SET_APP_START, std::bind(&MCPlugin::StartPlugin, this,
                                    std::placeholders::_1)},
          {SET_APP_STOP, std::bind(&MCPlugin::StopPlugin, this,
                                   std::placeholders::_1)} };

  bool auto_start_ = false;
  bool enable_vot_ = false;
  std::map<uint64_t, std::shared_ptr<horizon::vision::VotData_t>>
          cache_vot_data_;
  std::mutex mut_cache_;
  std::condition_variable cv_;
  uint64_t cache_len_limit_ = 10;
  std::vector<std::shared_ptr<std::thread>> feed_vo_thread_;
  int feed_vo_thread_num_ = 1;

  std::shared_ptr<std::thread> status_monitor_thread_ = nullptr;
  std::atomic_int unrecv_ap_count_;
  int unrecv_ap_count_max = 5;
  int OnGetAPImage(const x3::Image &image, uint64_t seq_id);
};

// int MCPlugin::cp_status_;
// int MCPlugin::log_level_;

}  // namespace mcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // APP_INCLUDE_PLUGIN_MCPLUGIN_MCPLUGIN_H_
