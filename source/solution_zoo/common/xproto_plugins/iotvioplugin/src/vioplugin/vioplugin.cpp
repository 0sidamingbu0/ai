/*
 * @Description: implement of vioplugin
 * @Author: fei.cheng@horizon.ai
 * @Date: 2019-08-26 16:17:25
 * @Author: songshan.gong@horizon.ai
 * @Date: 2019-09-26 16:17:25
 * @LastEditors: hao.tian@horizon.ai
 * @LastEditTime: 2019-10-16 15:35:08
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "hobotlog/hobotlog.hpp"
#include "vioplugin/vioplugin.h"
#include "vioplugin/vioproduce.h"

#include "utils/time_helper.h"

#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto_msgtype/hbipcplugin_data.h"

#include "xproto_msgtype/protobuf/pack.pb.h"
#include "xproto_msgtype/protobuf/x2.pb.h"

XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_IMAGE_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_DROP_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_DROP_IMAGE_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_FACE_PIC_IMAGE_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_INFO_IMAGE_MESSAGE)
namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

using horizon::vision::xproto::basic_msgtype::HbipcMessage;

int VioPlugin::OnGetHbipcResult(XProtoMessagePtr msg) {
  int ret = 0;
  std::string proto_str;
  static int fps = 0;
  // 耗时统计
  static auto lastTime = hobot::Timer::tic();
  static int frameCount = 0;

  ++frameCount;

  auto curTime = hobot::Timer::toc(lastTime);
  // 统计数据发送帧率
  if (curTime > 1000) {
    fps = frameCount;
    frameCount = 0;
    lastTime = hobot::Timer::tic();
    LOGE << "[HbipcPlugin] fps = " << fps;
  }

  auto hbipc_message = std::static_pointer_cast<HbipcMessage>(msg);
  x2::InfoMessage proto_info_message;
  proto_info_message.ParseFromString(hbipc_message->Serialize());
  if (proto_info_message.config__size() <= 0) {
    LOGE << "PB don't have config param";
    return false;
  }
  for (auto i = 0; i < proto_info_message.config__size(); ++i) {
    auto &params = proto_info_message.config_(i);
    if (params.type_() == "Detect") {
      LOGI << "Set CP detect params";
      for (auto j = 0; j < params.shield__size(); ++j) {
        box_t box;
        auto &shield = params.shield_(j);
        if (shield.type_() == "valid_zone" ||
            shield.type_() == "invalid_zone") {
          auto &point1 = shield.top_left_();
          auto &point2 = shield.bottom_right_();
          box.x1 = point1.x_();
          box.y1 = point1.y_();
          box.x2 = point2.x_();
          box.y2 = point2.y_();
          LOGI << shield.type_() << ":" << box.x1 << " " << box.y1 << " "
               << box.x2 << " " << box.y2;
          Shields_.emplace_back(box);
        } else {
          LOGE << "invalid zone type: " << shield.type_();
        }
      }
    }
  }
  return ret;
}

VioPlugin::VioPlugin(const std::string &path) {
  config_ = GetConfigFromFile(path);
  GetSubConfigs();
  HOBOT_CHECK(configs_.size() > 0);
}

void VioPlugin::GetSubConfigs() {
  HOBOT_CHECK(config_);
  if (config_->HasMember("config_number")) {
    vio_config_num_ = config_->GetIntValue("config_number");
  }
  HOBOT_CHECK(vio_config_num_ > 0);
  if (config_->HasMember("config_data")) {
    auto multi_configs = config_->GetSubConfig("config_data");
    int cnt = multi_configs->ItemCount();
    HOBOT_CHECK(vio_config_num_ <= cnt) << "error vio config number";
    for (int i = 0; i < vio_config_num_; i++) {
      configs_.push_back(multi_configs->GetSubConfig(i));
    }
  } else {
    configs_.push_back(config_);
  }
}

int VioPlugin::Init() {
  if (is_inited_)
    return 0;
  ClearAllQueue();
  for (auto config : configs_) {
    auto data_source_ = config->GetValue("data_source");
    auto vio_handle = VioProduce::CreateVioProduce(config, data_source_);
    HOBOT_CHECK(vio_handle);
    vio_handle->SetConfig(config);
    vio_handle->SetVioConfigNum(vio_config_num_);
    vio_produce_handles_.push_back(vio_handle);
  }
#ifdef USE_MC
  RegisterMsg(TYPE_APIMAGE_MESSAGE,
              std::bind(&VioPlugin::OnGetAPImage, this,
                        std::placeholders::_1));
#endif
  if (!is_sync_mode_) {
    // 注册智能帧结果
    RegisterMsg(TYPE_HBIPC_MESSAGE,
                std::bind(&VioPlugin::OnGetHbipcResult, this,
                          std::placeholders::_1));
    // 调用父类初始化成员函数注册信息
    XPluginAsync::Init();
  } else {
    LOGI << "Sync mode";
    // XPluginAsync::Init();
  }
  is_inited_ = true;
  return 0;
}

int VioPlugin::DeInit() {
  VioPipeManager &manager = VioPipeManager::Get();
  manager.Reset();
  vio_produce_handles_.clear();
  is_inited_ = false;
  XPluginAsync::DeInit();
  return 0;
}

VioPlugin::~VioPlugin() {}

Listener VioPlugin::SetListenerFunc() {
  auto send_frame = [&](const std::shared_ptr<VioMessage> input) {
    if (!input) {
      LOGE << "VioMessage is NULL, return";
      return -1;
    }

    if (!is_sync_mode_) {
      PushMsg(input);
    } else {
      if (input->type_ == TYPE_IMAGE_MESSAGE) {
        img_msg_queue_.push(input);
      } else if (input->type_ == TYPE_DROP_MESSAGE) {
        drop_msg_queue_.push(input);
      } else {
        LOGE << "received message with unknown type";
      }
    }
    return 0;
  };

  return send_frame;
}

int VioPlugin::Start() {
  int ret;
  if (is_running_) return 0;
  is_running_ = true;

  auto send_frame = SetListenerFunc();
  for (auto vio_handle : vio_produce_handles_) {
    vio_handle->SetListener(send_frame);
    ret = vio_handle->Start();
    if (ret < 0) {
      LOGF << "VioPlugin start failed, err: " << ret << std::endl;
      return -1;
    }
  }

  return 0;
}

int VioPlugin::CamStart() {
  int ret;
  if (is_running_) return 0;
  is_running_ = true;

  auto send_frame = SetListenerFunc();
  for (auto vio_handle : vio_produce_handles_) {
    if (vio_handle->vio_src_type_ == kHorizonVioTypeCam) {
      vio_handle->SetListener(send_frame);
      ret = vio_handle->Start();
      if (ret < 0) {
        LOGF << "VioPlugin start failed, err: " << ret << std::endl;
        return -1;
      }
    }
  }
  return 0;
}

int VioPlugin::FbStart() {
  int ret;
  if (is_running_) return 0;
  is_running_ = true;

  auto send_frame = SetListenerFunc();
  for (auto vio_handle : vio_produce_handles_) {
    if (vio_handle->vio_src_type_ == kHorizonVioTypeFb) {
      vio_handle->SetListener(send_frame);
      ret = vio_handle->Start();
      if (ret < 0) {
        LOGF << "VioPlugin start failed, err: " << ret << std::endl;
        return -1;
      }
    }
  }
  return 0;
}

int VioPlugin::Stop() {
  LOGW << "Enter vioplugin stop, vio_handle_num: "
    << vio_produce_handles_.size();
  if (!is_running_) return 0;
  is_running_ = false;
  ClearAllQueue();
  for (auto vio_handle : vio_produce_handles_) {
    LOGW << "Enter vio_handle stop...";
    vio_handle->Stop();
  }
  return 0;
}

int VioPlugin::CamStop() {
  if (!is_running_) return 0;
  is_running_ = false;
  ClearAllQueue();
  for (auto vio_handle : vio_produce_handles_) {
    if (vio_handle->vio_src_type_ == kHorizonVioTypeCam) {
      vio_handle->Stop();
    }
  }
  return 0;
}

int VioPlugin::FbStop() {
  if (!is_running_) return 0;
  is_running_ = false;
  ClearAllQueue();
  for (auto vio_handle : vio_produce_handles_) {
    if (vio_handle->vio_src_type_ == kHorizonVioTypeFb) {
      vio_handle->Stop();
    }
  }
  return 0;
}

std::shared_ptr<VioConfig> VioPlugin::GetConfigFromFile(
    const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    LOGF << "Open config file " << path << " failed";
    return nullptr;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  ifs.close();
  std::string content = ss.str();
  Json::Value value;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  JSONCPP_STRING error;
  std::shared_ptr<Json::CharReader> reader(builder.newCharReader());
  try {
    bool ret = reader->parse(content.c_str(), content.c_str() + content.size(),
                             &value, &error);
    if (ret) {
      auto config = std::shared_ptr<VioConfig>(new VioConfig(value));
      return config;
    } else {
      return nullptr;
    }
  } catch (std::exception &e) {
    return nullptr;
  }
}

XProtoMessagePtr VioPlugin::GetImage() {
  // compare img_msg_queue.head.frame_id with drop_msg_queue.head.frame_id
  // if former is greater than latter, pop from drop_msg_queue
  if (img_msg_queue_.size() == 0) {
    LOGI << "image message queue is empty";
    return nullptr;
  }
  auto img_msg = img_msg_queue_.peek();
  auto img_vio_msg = dynamic_cast<VioMessage *>(img_msg.get());
  auto img_msg_seq_id = img_vio_msg->sequence_id_;
  img_msg_queue_.pop();
  for (size_t i = 0; i < drop_msg_queue_.size(); ++i) {
    auto drop_msg = drop_msg_queue_.peek();
    auto drop_vio_msg = dynamic_cast<VioMessage *>(drop_msg.get());
    auto drop_msg_seq_id = drop_vio_msg->sequence_id_;
    if (drop_msg_seq_id != ++(img_msg_seq_id)) {
      if (drop_msg_seq_id < img_msg_seq_id) {
        LOGI << "Dropped msg id less than image msg id";
        drop_msg_queue_.pop();
      } else {
        LOGI << "Current image msg id: " << img_msg_seq_id
             << "dropped msg id: " << drop_msg_seq_id;
      }
    } else {
      // TODO(shiyu.fu): send drop message
      drop_msg_queue_.pop();
    }
  }
  return img_msg;
}

void VioPlugin::ClearAllQueue() {
  if (is_sync_mode_) {
    img_msg_queue_.clear();
    drop_msg_queue_.clear();
  }
}

#ifdef USE_MC
int VioPlugin::OnGetAPImage(XProtoMessagePtr msg) {
  for (auto &viosp : vio_produce_handles_) {
    std::string param = "success";
    if (0 != viosp->OnGetAPImage(msg)) {
      param = "fail";
    }
    // push response msg
    auto vio_resp_msg = std::make_shared<InfoVioMessage>();
    vio_resp_msg->sequence_id_ =
            std::static_pointer_cast<APImageMessage>(msg)->sequence_id_;
    vio_resp_msg->param_ = param;
    LOGD << "push msg " << vio_resp_msg->type()
         << " " << vio_resp_msg->sequence_id_;
    PushMsg(vio_resp_msg);
  }
  return 0;
}
#endif

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
