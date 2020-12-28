/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Fei cheng
 * @Mail: fei.cheng@horizon.ai
 * @Date: 2019-09-14 20:38:52
 * @Version: v0.0.1
 * @Brief: mcplugin impl based on xproto.
 * @Last Modified by: Fei cheng
 * @Last Modified time: 2019-09-14 22:41:30
 */

#include "mcplugin/mcplugin.h"

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "horizon/vision_type/vision_type_util.h"
#include "mcplugin/mcmessage.h"
#include "smartplugin/smartplugin.h"
#include "utils/executor.hpp"
#include "utils/votmodule.h"
#include "uvcplugin/uvcplugin.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto_msgtype/protobuf/x3.pb.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace mcplugin {

XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_MC_UPSTREAM_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_MC_DOWMSTREAM_MESSAGE)

using horizon::vision::xproto::smartplugin::CustomSmartMessage;

bool MCMessage::IsMessageValid(ConfigMessageMask mask) {
  return message_mask_ & mask ? true : false;
}

MCPlugin::MCPlugin(const std::string& config_file) {
  config_file_ = config_file;
  LOGI << "MC config file:" << config_file_;
  Json::Value cfg_jv;
  std::ifstream infile(config_file_);
  if (infile) {
    infile >> cfg_jv;
    config_.reset(new JsonConfigWrapper(cfg_jv));
    ParseConfig();
  } else {
    LOGE << "open mc config fail";
  }
}

void MCPlugin::ParseConfig() { LOGI << "Parse Config Done!"; }

int MCPlugin::Init() {
  LOGI << "MCPlugin INIT";
  RegisterMsg(TYPE_TRANSPORT_MESSAGE, std::bind(&MCPlugin::OnGetUvcResult, this,
                                                std::placeholders::_1));
  RegisterMsg(TYPE_SMART_MESSAGE, std::bind(&MCPlugin::OnGetSmarterResult, this,
                                            std::placeholders::_1));
  RegisterMsg(TYPE_IMAGE_MESSAGE, std::bind(&MCPlugin::OnGetVioResult, this,
                                            std::placeholders::_1));
  RegisterMsg(TYPE_DROP_IMAGE_MESSAGE, std::bind(&MCPlugin::OnGetVioResult,
                                                 this, std::placeholders::_1));
  return XPluginAsync::Init();
}

std::shared_ptr<HorizonVisionImageFrame>
ConstructVotImgData(const std::shared_ptr<VioMessage>& vio_) {
  HOBOT_CHECK(vio_);
  // copy img
  auto sp_img = std::shared_ptr<HorizonVisionImageFrame>(
          new HorizonVisionImageFrame,
          [&](HorizonVisionImageFrame* p){
              if (p) {
                if (p->image.data) {
                  std::free(p->image.data);
                  p->image.data = NULL;
                }
                delete p;
                p = NULL;
              }
          });
  // copy img
  sp_img->frame_id = vio_->sequence_id_;
  const auto& pym_info = vio_->image_.front()->down_scale[0];
  sp_img->image.data_size =
          pym_info.width * pym_info.height * 3 / 2;
  sp_img->image.height =
          pym_info.height;
  sp_img->image.width =
          pym_info.width;
  sp_img->image.data =
          static_cast<uint8_t *>(
                  std::calloc(sp_img->image.data_size,
                              sizeof(uint8_t)));
  memcpy(sp_img->image.data,
         reinterpret_cast<char*>(pym_info.y_vaddr),
         pym_info.width * pym_info.height);
  memcpy(sp_img->image.data
         + pym_info.width * pym_info.height,
         reinterpret_cast<char*>(pym_info.c_vaddr),
         pym_info.width * pym_info.height / 2);

  return sp_img;
}

std::shared_ptr<std::string>
ConstructVotSmartData(const std::shared_ptr<SmartMessage>& smart_) {
  HOBOT_CHECK(smart_);
  auto smart_msg = dynamic_cast<CustomSmartMessage *>(smart_.get());
  HOBOT_CHECK(smart_msg);

  smart_msg->SetAPMode(true);
  std::shared_ptr<std::string> sp_smart_pb_ = std::make_shared<std::string>(
          smart_msg->Serialize(1920, 1080, 1920, 1080));
  HOBOT_CHECK(sp_smart_pb_);

  return sp_smart_pb_;
}

int MCPlugin::Start() {
  is_running_ = true;
  cp_status_ = CP_READY;

  if (config_ && config_->HasKey("enable_auto_start") &&
      config_->GetBoolValue("enable_auto_start")) {
    LOGI << "enable auto start";
    auto_start_ = true;

    if (config_->HasKey("enable_vot") &&
        config_->GetBoolValue("enable_vot")) {
      LOGI << "enable vot";
      enable_vot_ = true;
    }

    for (auto i = PluginContext::Instance().basic_plugin_cnt;
         i < PluginContext::Instance().plugins.size(); ++i) {
      PluginContext::Instance().plugins[i]->Init();
      PluginContext::Instance().plugins[i]->Start();
    }

    //  start vot
    if (enable_vot_) {
        horizon::vision::VotModule::Instance()->Start();
      //  feed vo
      auto task = [this] () {
          while (is_running_) {
            std::map<uint64_t,
                 std::shared_ptr<horizon::vision::VotData_t>>::iterator front;
            std::shared_ptr<horizon::vision::VotData_t> vot_feed = nullptr;

            std::unique_lock<std::mutex> lg(mut_cache_);
            cv_.wait(lg, [this] () {
                return !is_running_ || !cache_vot_data_.empty();
            });
            if (!is_running_) {
              cache_vot_data_.clear();
              break;
            }

            if (cache_vot_data_.empty()) {
              continue;
            }

            front = cache_vot_data_.begin();

            if (((!front->second->is_drop_frame_ &&
                    front->second->sp_img_ &&
                    front->second->sp_smart_pb_) ||
                    (front->second->is_drop_frame_ &&
                            front->second->sp_img_))) {
              vot_feed = front->second;
              cache_vot_data_.erase(front);
              lg.unlock();
              horizon::vision::VotModule::Instance()->Input(vot_feed);
              continue;
            }

            LOGV << "cache_vio_smart_ size:" << cache_vot_data_.size();

            if (cache_vot_data_.size() >= cache_len_limit_) {
              // exception occurred
              vot_feed = front->second;
              cache_vot_data_.erase(front);
              lg.unlock();
              horizon::vision::VotModule::Instance()->Input(vot_feed);
              continue;
            }
          }

          LOGD << "thread  exit";
      };
      for (int i = 0; i < feed_vo_thread_num_; i++) {
        feed_vo_thread_.emplace_back(std::make_shared<std::thread>(task));
      }
    }
  } else {
    LOGI << "unable auto start";
    auto_start_ = false;
  }

  if (!status_monitor_thread_) {
    status_monitor_thread_ = std::make_shared<std::thread>(
            [this] () {
                unrecv_ap_count_ = 0;
                while (is_running_) {
                  if (CP_WORKING == cp_status_) {
                    unrecv_ap_count_++;
                    if (unrecv_ap_count_ > unrecv_ap_count_max) {
                      LOGW << "unrecv_ap_count_:" << unrecv_ap_count_
                           << " exceeds limit:" << unrecv_ap_count_max;
                      StopPlugin(0);
                    }
                  }
                  std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            });
  }

  LOGD << "mc start";
  return 0;
}

int MCPlugin::Stop() {
  LOGD << "mc stop...";
  is_running_ = false;
  cp_status_ = CP_READY;

  LOGD << "feed thread stop start";
  cv_.notify_all();
  for (auto thread : feed_vo_thread_) {
    if (thread) {
      thread->join();
    }
  }
  feed_vo_thread_.clear();
  LOGD << "feed thread stop done";

  status_monitor_thread_->join();
  status_monitor_thread_ = nullptr;

  {
    std::unique_lock<std::mutex> lg(mut_cache_);
    cache_vot_data_.clear();
  }
  LOGD << "cache_vio_smart_ clear";

  for (auto i = PluginContext::Instance().basic_plugin_cnt;
       i < PluginContext::Instance().plugins.size(); ++i) {
    PluginContext::Instance().plugins[i]->Stop();
    PluginContext::Instance().plugins[i]->DeInit();
  }
  LOGD << "plugin stop done";

  if (auto_start_ && enable_vot_) {
    LOGD << "vot stop";
      horizon::vision::VotModule::Instance()->Stop();
  }
  LOGD << "mc stop end";
  return 0;
}

int MCPlugin::StartPlugin(uint64 msg_id) {
  int ret = 0;
  if (cp_status_ != CP_WORKING) {
    cp_status_ = CP_STARTING;
    for (auto i = PluginContext::Instance().basic_plugin_cnt;
         i < PluginContext::Instance().plugins.size(); ++i) {
      PluginContext::Instance().plugins[i]->Init();
      ret = PluginContext::Instance().plugins[i]->Start();
      if (ret) {
        for (auto j = PluginContext::Instance().basic_plugin_cnt; j < i; ++j) {
          PluginContext::Instance().plugins[i]->Stop();
          PluginContext::Instance().plugins[i]->DeInit();
        }
        LOGE << "StartPlugin failed, index:" << i;
        break;
      }
    }
  }

  auto rsp_data = std::make_shared<SmarterConfigData>(ret == 0);
  auto mcmsg = std::make_shared<MCMessage>(SET_APP_START, MESSAGE_TO_ADAPTER,
                                           rsp_data, "up", msg_id);
  PushMsg(mcmsg);

  if (ret) {
    cp_status_ = CP_ABNORMAL;
  } else {
    cp_status_ = CP_WORKING;
  }

  LOGD << "StartPlugin done msg_id:" << msg_id;
  return 0;
}

int MCPlugin::StopPlugin(uint64 msg_id) {
  if (cp_status_ != CP_READY) {
    for (auto i = PluginContext::Instance().basic_plugin_cnt;
         i < PluginContext::Instance().plugins.size(); ++i) {
      LOGI << "stop plugin NO. " << i
           << "  " << PluginContext::Instance().plugins[i]->desc();
      PluginContext::Instance().plugins[i]->Stop();
      PluginContext::Instance().plugins[i]->DeInit();
    }
  }

  auto rsp_data = std::make_shared<SmarterConfigData>(true);
  auto mcmsg = std::make_shared<MCMessage>(SET_APP_STOP, MESSAGE_TO_ADAPTER,
                                           rsp_data, "up", msg_id);
  PushMsg(mcmsg);
  cp_status_ = CP_READY;
  LOGD << "StopPlugin done";
  return 0;
}

int MCPlugin::GetCPStatus() { return cp_status_; }

int MCPlugin::GetCPLogLevel() { return log_level_; }

void MCPlugin::ConstructMsgForCmd(const x3::InfoMessage& InfoMsg,
                                  uint64 msg_id) {
  ConfigMessageType type;

  if (!InfoMsg.has_command_()) {
    LOGE << "msg has no cmd";
    return;
  }

  auto cmd = InfoMsg.command_();
  LOGI << "cmd:" << cmd.order_();
  if (cmd.order_() == x3::Command_Order::Command_Order_StartX2) {
    type = SET_APP_START;
  } else if (cmd.order_() == x3::Command_Order::Command_Order_StopX2) {
    type = SET_APP_STOP;
  } else {
    LOGE << "unsupport cmd order: " << cmd.order_();
    return;
  }

  if (cmd_func_map.find(type) != cmd_func_map.end()) {
    Executor::GetInstance()->AddTask(
        [this, msg_id, type]() -> int { return cmd_func_map[type](msg_id); });
  } else {
    LOGE << "No support CMD found!";
  }
}

int MCPlugin::OnGetAPImage(const x3::Image &image, uint64_t seq_id) {
  auto hg_msg = std::make_shared<APImageMessage>(
      image.buf_(), image.type_(),
      image.width_(), image.height_(),
      seq_id);
  PushMsg(hg_msg);
  return kHorizonVisionSuccess;
}

int MCPlugin::OnGetUvcResult(const XProtoMessagePtr& msg) {
  auto uvc_msg = std::static_pointer_cast<TransportMessage>(msg);
  LOGD << "recv uvc ";
  x3::MessagePack pack_msg;
  x3::InfoMessage InfoMsg;
  x3::Image image;
  unrecv_ap_count_ = 0;
  auto pack_msg_parse = pack_msg.ParseFromString(uvc_msg->proto_);
  if (pack_msg_parse &&
      InfoMsg.ParseFromString(pack_msg.content_())) {
    if (InfoMsg.has_command_()) {
      auto msg_type = InfoMsg.command_().order_();
      LOGI << "msg_type is " << msg_type;
      if (x3::Command_Order_StartX2 == msg_type ||
          x3::Command_Order_StopX2 == msg_type) {
        uint64 msg_id = 0;
        if (pack_msg.has_addition_() && pack_msg.addition_().has_frame_()) {
          msg_id = pack_msg.addition_().frame_().sequence_id_();
          LOGD << "msg_id: " << msg_id;
        }
        ConstructMsgForCmd(InfoMsg, msg_id);
      } else {
        LOGE << "Msg sub Type Error, type is " << msg_type;
        return -1;
      }
    }

    if (InfoMsg.has_status_()) {
      auto msg_type = InfoMsg.status_().GetTypeName();
      LOGI << "msg_type is " << msg_type;
      auto msg_stat = InfoMsg.status_().run_status_();
      LOGI << "msg_stat is " << static_cast<int>(msg_stat)
           << "  ts " << InfoMsg.status_().timestamp_();
      if (x3::Status::Normal == msg_stat) {
      }
    }
  } else if (pack_msg_parse && pack_msg.has_addition_() &&
      pack_msg.addition_().has_frame_() &&
      image.ParseFromString(pack_msg.content_())) {
    return OnGetAPImage(image, pack_msg.addition_().frame_().sequence_id_());
  } else {
    LOGE << "parse msg fail";
    LOGE << "pack_msg.has_addition_() " << pack_msg.has_addition_();
    LOGE << "pack_msg.addition_().has_frame_() "
    << pack_msg.addition_().has_frame_();
    LOGE << "pack_msg.addition_().frame id: " <<
    pack_msg.addition_().frame_().sequence_id_();
    LOGE << "image ParseFromString " <<
    image.ParseFromString(pack_msg.content_());
    return kHorizonVisionFailure;
  }

  return kHorizonVisionSuccess;
}

int MCPlugin::OnGetSmarterResult(const XProtoMessagePtr& msg) {
  if (!is_running_ || !auto_start_ || !enable_vot_) {
    return 0;
  }
  auto smartmsg = std::static_pointer_cast<SmartMessage>(msg);
  HOBOT_CHECK(smartmsg);
  auto frame_id = smartmsg->frame_id;
  LOGD << "smart frame_id:" << frame_id;
  auto sp_smart = ConstructVotSmartData(smartmsg);
  std::lock_guard<std::mutex> lg(mut_cache_);

  if (cache_vot_data_.find(frame_id) == cache_vot_data_.end()) {
    // recv vio msg faster than smart msg
    // if run here, noting vio plugin do not send drop frame
    auto cache_vio_smart = std::make_shared<horizon::vision::VotData_t>();
    cache_vio_smart->sp_smart_pb_ = sp_smart;
    cache_vot_data_[frame_id] = cache_vio_smart;
  } else {
    // filter ware plugin msg
    if (!cache_vot_data_[frame_id]->sp_smart_pb_) {
      cache_vot_data_[frame_id]->sp_smart_pb_ = sp_smart;
    }
  }
  cv_.notify_one();
  LOGV << "cache_vio_smart_ size:" << cache_vot_data_.size();

  return 0;
}

int MCPlugin::OnGetVioResult(const XProtoMessagePtr& msg) {
  if (!is_running_ || !auto_start_ || !enable_vot_) {
    return 0;
  }

  auto valid_frame = std::static_pointer_cast<VioMessage>(msg);
  if (valid_frame == nullptr) {
    LOGE << "valid_frame is null";
    return -1;
  } else {
    if (valid_frame->image_.empty()) {
      LOGE << "valid_frame->image_.empty() is empty";
      return 1;
    } else {
      if (valid_frame->image_.front() == nullptr) {
        LOGE << "valid_frame->image_.front()is empty";
        return -1;
      }
    }
  }
  HOBOT_CHECK(valid_frame && !valid_frame->image_.empty() &&
              valid_frame->image_.front());
  LOGD << "type:" << valid_frame->type()
       << " seq id:" << valid_frame->sequence_id_
       << " ts:" << valid_frame->time_stamp_;
  auto frame_id = valid_frame->image_.front()->frame_id;
  LOGI << "vio frame_id:" << frame_id;
  bool is_drop_frame = false;
  if (TYPE_IMAGE_MESSAGE == valid_frame->type()) {
    is_drop_frame = false;
  } else if (TYPE_DROP_IMAGE_MESSAGE == valid_frame->type()) {
    is_drop_frame = true;
  } else {
    LOGE << "invalid type " << valid_frame->type();
    is_drop_frame = false;
  }

  std::shared_ptr<HorizonVisionImageFrame> sp_img = nullptr;
  sp_img = ConstructVotImgData(valid_frame);
  std::lock_guard<std::mutex> lg(mut_cache_);

  if (cache_vot_data_.find(frame_id) == cache_vot_data_.end()) {
    auto cache_vio_smart = std::make_shared<horizon::vision::VotData_t>();
    cache_vio_smart->is_drop_frame_ = is_drop_frame;
    cache_vio_smart->sp_img_ = sp_img;
    cache_vot_data_[frame_id] = cache_vio_smart;
  } else {
    cache_vot_data_[frame_id]->is_drop_frame_ = is_drop_frame;
    cache_vot_data_[frame_id]->sp_img_ = sp_img;
  }

  cv_.notify_one();
  LOGV << "cache_vio_smart_ size:" << cache_vot_data_.size();
  return 0;
}

std::string MCMessage::Serialize() {
  LOGD << "serialize msg_id_:" << msg_id_;
  x3::MessagePack pack;
  pack.set_flow_(x3::MessagePack_Flow_CP2AP);
  pack.set_type_(x3::MessagePack_Type::MessagePack_Type_kXConfig);
  pack.mutable_addition_()->mutable_frame_()->set_sequence_id_(msg_id_);

  x3::InfoMessage info;
  auto ack =
      std::dynamic_pointer_cast<SmarterConfigData>(message_data_)->status_
          ? x3::Response_Ack::Response_Ack_Success
          : x3::Response_Ack::Response_Ack_Fail;
  info.mutable_response_()->set_ack_(ack);
  pack.set_content_(info.SerializeAsString());

  return pack.SerializeAsString();
}
}  // namespace mcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
