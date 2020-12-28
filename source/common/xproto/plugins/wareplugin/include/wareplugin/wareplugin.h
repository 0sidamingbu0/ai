/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail:
 * @Date: 2020-11-02 20:38:52
 * @Version: v0.0.1
 * @Last Modified by:
 * @Last Modified time: 2020-11-02 20:38:52
 */

#include <string>
#include <memory>
#include <utility>
#include "smartplugin/smartplugin.h"
#include "wareplugin/waredb.h"
#include "wareplugin/utils/jsonConfigWrapper.h"
#include "xproto_msgtype/uvcplugin_data.h"
#include "xproto_msgtype/protobuf/x3ware.pb.h"
#ifndef INCLUDE_WAREPLUGIN_WAREPLUGIN_H_
#define INCLUDE_WAREPLUGIN_WAREPLUGIN_H_
namespace horizon {
namespace vision {
namespace xproto {
namespace wareplugin {

using horizon::vision::xproto::basic_msgtype::TransportMessage;

using SnapshotInfoXStreamBaseData =
hobot::vision::SnapshotInfo<xstream::BaseDataPtr>;
using SnapshotInfoXStreamBaseDataPtr =
std::shared_ptr<SnapshotInfoXStreamBaseData>;
using XStreamSnapshotInfo =
xstream::XStreamData<SnapshotInfoXStreamBaseDataPtr>;
using XStreamSnapshotInfoPtr = std::shared_ptr<XStreamSnapshotInfo>;
struct WareSmartMessage : public smartplugin::CustomSmartMessage {
  explicit WareSmartMessage(
      xstream::OutputDataPtr out) : CustomSmartMessage(out) {
  }
  explicit WareSmartMessage(const CustomSmartMessage& smart_message) :
      CustomSmartMessage(smart_message) { }
};

struct WareLibRecordMessage : public WareSmartMessage {
  explicit WareLibRecordMessage(x3ware::WareMessage&& waremessage,
      uint64_t ware_seq_id) :  WareSmartMessage(nullptr),
      ware_seq_id_(ware_seq_id) {
    ware_msg_fromap_ = std::forward<x3ware::WareMessage>(waremessage);
  }
  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) override;

 private:
  x3ware::WareMessage ware_msg_fromap_;
  uint64_t ware_seq_id_;

 private:
  void UnknownMsgType(x3ware::WareMessage *ware_msg_res);
  int WareSearchRecord(x3ware::WareMessage *ware_msg_res);
  int WareTableOperation(x3ware::WareMessage_Oper msg_oper,
      x3ware::WareMessage *ware_msg_res);
  int WareRecordOperation(x3ware::WareMessage_Oper msg_oper,
      x3ware::WareMessage *ware_msg_res);
};

struct SnapSmartMessage : public WareSmartMessage {
  explicit SnapSmartMessage(
      xstream::OutputDataPtr out) : WareSmartMessage(out) {
  }
  explicit SnapSmartMessage(const CustomSmartMessage& smart_message) :
      WareSmartMessage(smart_message) { }

  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) override;

  void SetRecognizeMode(bool is_recognize) {
    is_recognize_ = is_recognize;
  }
  void SetAddRecordMode(bool is_add_record) {
    is_add_record_ = is_add_record;
  }

 private:
  void ConvertSnapShotOutput(
      const xstream::BaseDataVector *targets_snapshots,
      const xstream::BaseDataVector *targets_face_features,
      float x_ratio, float y_ratio,
      x3::CaptureFrameMessage *capture_msg);
  void ConvertTargetInfo(const xstream::BaseDataVector *target_snaps,
                         const xstream::BaseDataVector *target_features,
                         float x_ratio, float y_ratio,
                         x3::CaptureTarget *capture_targets);
  void ConvertRecognize(const xstream::BaseDataPtr face_feature,
                        x3::DBResult *capture_dbresult);
  void ConvertSnapShotInfo(SnapshotInfoXStreamBaseDataPtr sp_snapshot,
                           const xstream::BaseDataPtr face_feature,
                           float x_ratio, float y_ratio,
                           x3::Capture *capture_target);

 private:
  bool is_recognize_ = false;
  bool is_add_record_ = false;
};

class WarePlugin : public XPluginAsync {
 public:
  WarePlugin() = default;
  explicit WarePlugin(const std::string& config_file);

  ~WarePlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;
  int DeInit() override;
  std::string desc() const { return "WarePlugin"; }

 private:
  int OnGetSmarterResult(const XProtoMessagePtr& msg);
  void ParseConfig();

  std::string config_file_;
  std::shared_ptr<JsonConfigWrapper> config_;

 private:
  bool is_recognize_ = false;
  bool is_add_record_ = false;
  float similar_thres_ = 0.74;
  bool isinit_ = false;
  int OnGetAPMessage(const XProtoMessagePtr &msg);
};

}   //  namespace wareplugin
}   //  namespace xproto
}   //  namespace vision
}   //  namespace horizon

#endif  //  INCLUDE_WAREPLUGIN_WAREPLUGIN_H_
