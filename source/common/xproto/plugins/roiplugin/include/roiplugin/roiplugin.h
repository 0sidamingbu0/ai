/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: xue.liang
 * @Mail: xue.liang@horizon.ai
 * @Date: 2020-12-15 20:38:52
 * @Version: v0.0.1
 * @Brief: roiplugin impl based on xpp.
 * @Last Modified by: xue.liang
 * @Last Modified time: 2020-12-16 22:41:30
 */

#ifndef APP_INCLUDE_PLUGIN_ROIPLUGIN_ROIPLUGIN_H_
#define APP_INCLUDE_PLUGIN_ROIPLUGIN_ROIPLUGIN_H_

#include <future>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hobotlog/hobotlog.hpp"
#include "hobotxsdk/xstream_sdk.h"
#include "horizon/vision_type/vision_type_util.h"
#include "roiplugin/roi_common.h"
#include "utils/jsonConfigWrapper.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto/msgtype/include/xproto_msgtype/roiplugin_data.h"
#include "xproto/msgtype/include/xproto_msgtype/smartplugin_data.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto/utils/singleton.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace roiplugin {
using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;

class RoiPlugin : public XPluginAsync {
 public:
  RoiPlugin() = default;
  explicit RoiPlugin(const std::string& config_file);

  ~RoiPlugin() = default;
  int Init() override;
  int DeInit() override;
  int Start() override;
  int Stop() override;

 private:
  int StartPlugin(uint64_t msg_id);
  int StopPlugin(uint64_t msg_id);
  int OnGetSmarterResult(const XProtoMessagePtr& msg);
  int OnGetVioResult(const XProtoMessagePtr& msg);
  int ParseConfig();
  void ConvertSmartToRoiData(xstream::OutputDataPtr xstream_out,
                             const uint64_t frame_id, const uint64_t pts);

 private:
  std::atomic<bool> is_running_;
  std::string config_file_;
  Json::Value config_;
  bool auto_start_ = false;
  bool enable_vot_ = false;
  bool enable_intelligent_tracking_ = false;
  bool is_init_ = false;

  enum SmartType { SMART_FACE, SMART_BODY, SMART_VEHICLE };
  SmartType smart_type_ = SMART_BODY;
};

}  // namespace roiplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // APP_INCLUDE_PLUGIN_ROIPLUGIN_ROIPLUGIN_H_
