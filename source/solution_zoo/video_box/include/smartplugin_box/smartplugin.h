/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Songshan Gong
 * @Mail: songshan.gong@horizon.ai
 * @Date: 2019-08-04 02:41:22
 * @Version: v0.0.1
 * @Brief: smartplugin declaration
 * @Last Modified by: Songshan Gong
 * @Last Modified time: 2019-09-30 00:45:01
 */

#ifndef INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_
#define INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_

#include <memory>
#include <string>
#include <vector>

#include <thread>
#include <unordered_map>
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"

#include "hobotxsdk/xstream_sdk.h"
#include "smartplugin_box/runtime_monitor.h"
#include "smartplugin_box/smart_config.h"
#include "smartplugin_box/traffic_info.h"
#include "votmodule.h"
#include "xproto_msgtype/smartplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin_multiplebox {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;

using horizon::vision::xproto::basic_msgtype::SmartMessage;
using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;

using horizon::vision::xproto::smartplugin::VehicleInfo;
// using horizon::vision::xproto::smartplugin::NoMotorVehicleInfo;
// using horizon::vision::xproto::smartplugin::PersonInfo;

struct VehicleSmartMessage
        : public horizon::vision::xproto::basic_msgtype::SmartMessage {
 public:
  std::string Serialize() override { return ""; }
//  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h);
//  void Serialize_Print();
 public:
//  int camera_type;
  std::vector<VehicleInfo> vehicle_infos;
//  std::vector<NoMotorVehicleInfo> nomotor_infos;
//  std::vector<PersonInfo> person_infos;
//  std::vector<uint64_t> lost_track_ids;
};


struct CustomSmartMessage : SmartMessage {
  explicit CustomSmartMessage(xstream::OutputDataPtr out) : smart_result(out) {
    type_ = TYPE_SMART_MESSAGE;
  }
  std::string Serialize() { return ""; };
  void Serialize_Print(Json::Value &root){};

private:
  xstream::OutputDataPtr smart_result;
};

#define TYPE_FEATURE_MESSAGE "XPLUGIN_FEATURE_MESSAGE"

class SmartFeatureMessage : public XProtoMessage {
 public:
  explicit SmartFeatureMessage(std::shared_ptr<FeatureFrameMessage> features) {
    type_ = TYPE_FEATURE_MESSAGE;
    features_ = features;
  }
  ~SmartFeatureMessage() = default;
  std::shared_ptr<FeatureFrameMessage> GetMessageData() { return features_; }
  std::string& GetMessageType() { return message_type_; }
  void SetMessageType(std::string msg_type) { message_type_ = msg_type; }
  std::string Serialize() override { return "no need serial";}

 private:
  std::shared_ptr<FeatureFrameMessage> features_;
  std::string message_type_;
};

using SmartFeatureMessagePtr = std::shared_ptr<SmartFeatureMessage>;

class SmartPlugin : public XPluginAsync {
 public:
  SmartPlugin() = default;
  explicit SmartPlugin(const std::string &smart_config_file);

  void SetConfig(const std::string &config_file) {
    smart_config_file_ = config_file;
  }

  ~SmartPlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;

 private:
  int Feed(XProtoMessagePtr msg);
  void OnCallback(xstream::OutputDataPtr out);
  int FeedPic(XProtoMessagePtr msg);
  void OnCallbackPic(xstream::OutputDataPtr out);
  void OnCallbackFeature(xstream::OutputDataPtr out);
  int FeedRecog(XProtoMessagePtr msg);

  void ParseConfig();
  void GetRtspConfigFromFile(const std::string &path);
  void GetDisplayConfigFromFile(const std::string &path);

  std::vector<std::shared_ptr<XStreamSDK>> sdk_;
  std::shared_ptr<XStreamSDK> pic_sdk_;
  std::shared_ptr<XStreamSDK> feature_sdk_;

  std::string smart_config_file_;
  std::string rtsp_config_file_;
  std::string display_config_file_;
  std::shared_ptr<RuntimeMonitor> monitor_;
  std::shared_ptr<JsonConfigWrapper> config_;
  std::string xstream_workflow_cfg_file_;
  std::string xstream_workflow_cfg_pic_file_;
  std::string xstream_workflow_cfg_feature_file_;
  bool enable_profile_{false};
  std::string profile_log_file_;
  bool result_to_json{false};
  Json::Value root;

  bool running_venc_1080p_ = false;
  bool running_venc_720p_ = false;
  bool running_vot_ = true;
  bool encode_smart_ = true;
  bool enable_recog_{false};

  int channel_num_ = 0;
  int display_mode_ = 0;

  smart_vo_cfg_t smart_vo_cfg_;
  bool running_;
  std::unordered_map<uint64_t, std::unordered_map<uint64_t,
      std::shared_ptr<RecogResult_t>>> recog_cache_;
  std::mutex cache_mtx_;
  bool run_smart_ = true;
  bool draw_smart_ = true;
  bool draw_real_time_video_ = true;
};

}  // namespace smartplugin_multiplebox
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_
