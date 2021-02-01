/*
 * @Description: implement of CommonGdcPlugin.cpp
 * @Author: xudong.du@horizon.ai
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#include <cstdlib>
#include <cstring>
#include <utility>
#include <string>
#include <memory>
#include <fstream>

// Horizon Header
#include "hobotlog/hobotlog.hpp"
#include "json/json.h"
#include "horizon/vision_type/vision_type.hpp"

// custom header
#include "commongdcplugin/commongdcplugin.h"
#include "utils/time_helper.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_GDC_MESSAGE)

using std::chrono::milliseconds;

CommonGdcPlugin::CommonGdcPlugin(std::string cfg_file) {
  config_file_ = cfg_file;
}

CommonGdcPlugin::~CommonGdcPlugin() {}

int CommonGdcPlugin::ParseConfig() {
  std::ifstream ifs(config_file_);
  if (!ifs.is_open()) {
    LOGE << "Open config file " << config_file_ << " failed";
    return kHorizonVisionFailure;
  }
  ifs >> config_;
  ifs.close();

  bool gdc_enable;
  bool all_in_one_vio;
  int data_source_num;
  int pym_layer;
  std::vector<int> channel2direction;
  std::vector<std::string> gdc_file_list;

  auto value_js = config_["gdc_enable"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: gdc_enable";
    gdc_enable = false;
  } else {
    gdc_enable = value_js.asBool();
  }

  value_js = config_["all_in_one_vio"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: all_in_one_vio";
    all_in_one_vio = false;
  } else {
    all_in_one_vio = value_js.asBool();
  }

  value_js = config_["data_source_num"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: data_source_num";
    data_source_num = 0;
  } else {
    data_source_num = value_js.asInt();
  }

  value_js = config_["pym_layer"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: pym_layer";
    pym_layer = 0;
  } else {
    pym_layer = value_js.asInt();
  }

  value_js = config_["channel2direction"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: channel2direction";
  } else {
    for (Json::Value::ArrayIndex i = 0; i < value_js.size(); ++i) {
      int chn2d = value_js[i].asInt();
      channel2direction.push_back(chn2d);
    }
  }

  value_js = config_["gdc_file_path"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: gdc_file_path";
  } else {
    for (Json::Value::ArrayIndex i = 0; i < value_js.size(); ++i) {
      std::string gdc_file = value_js[i].asString();
      gdc_file_list.push_back(gdc_file);
    }
  }

  auto channel2direction_size = channel2direction.size();
  auto gdc_file_list_size = gdc_file_list.size();
  LOGI << "Roiplugin Parse Config Done"
    << " gdc_enable: " << gdc_enable
    << " all_in_one_vio: " << all_in_one_vio
    << " data_source_num: " << data_source_num
    << " pym_layer: " << pym_layer
    << " channel2direction size: " << channel2direction_size;
  for (size_t i = 0; i < channel2direction_size; i++) {
    LOGI << "channel2direction index: " << i
      << " value: " << channel2direction[i];
  }
  for (size_t i = 0; i < gdc_file_list_size; i++) {
    LOGI << "gdc_file_list_size index: " << i
      << " value: " << gdc_file_list[i];
  }

  gdc_cfg_ = std::make_shared<GdcConfig>();
  gdc_cfg_->gdc_enable = gdc_enable;
  gdc_cfg_->all_in_one_vio = all_in_one_vio;
  gdc_cfg_->data_source_num = data_source_num;
  gdc_cfg_->pym_layer = pym_layer;
  gdc_cfg_->channel2direction = channel2direction;
  gdc_cfg_->gdc_file_path = gdc_file_list;
  GdcManager::Instance()->SetConfig(gdc_cfg_);

  return 0;
}

int CommonGdcPlugin::Init() {
  int ret = -1;
  LOGI << "Enter CommonGdcPlugin Init";
  ParseConfig();
  ret = GdcManager::Instance()->Init();
  HOBOT_CHECK(ret == 0) << "CommonGdcPlugin init falied";
  // 注册智能帧结果
  RegisterMsg(TYPE_IMAGE_MESSAGE, std::bind(&CommonGdcPlugin::OnGetVioMessage,
                                            this, std::placeholders::_1));
  gdc_work_thread_.CreatThread(1);
  is_stop_ = false;
  return 0;
}

int CommonGdcPlugin::DeInit() {
  int ret = -1;
  LOGI << "CommonGdcPlugin DeInit";
  ret = GdcManager::Instance()->DeInit();
  HOBOT_CHECK(ret == 0) << "CommonGdcPlugin deinit falied";
  is_stop_ = true;
  return 0;
}

int CommonGdcPlugin::Start() {
  int ret = -1;
  LOGI << "CommonGdcPlugin Start";
  ret = GdcManager::Instance()->Start();
  HOBOT_CHECK(ret == 0) << "CommonGdcPlugin start falied";
  return 0;
}

int CommonGdcPlugin::Stop() {
  int ret = -1;
  LOGI << "CommonGdcPlugin Stop";
  ret = GdcManager::Instance()->Stop();
  HOBOT_CHECK(ret == 0) << "CommonGdcPlugin stop falied";
  is_stop_ = true;
  return 0;
}

void CommonGdcPlugin::GetGdcResult(const XProtoMessagePtr msg) {
  int ret = -1;
  auto vio_message = std::static_pointer_cast<VioMessage>(msg);
  std::vector<std::shared_ptr<PymImageFrame>> &pym_images = vio_message->image_;
  auto image_num = pym_images.size();
  HOBOT_CHECK(image_num != 0);
  bool is_package = gdc_cfg_->all_in_one_vio;
  int chn_id = pym_images[0]->channel_id;
  std::vector<std::shared_ptr<SrcImageFrame>> gdc_images;
  if (is_package == true) {
    ret = GdcManager::Instance()->MultiInput(pym_images);
    if (ret) {
      LOGE << "CommonGdcPlugin multi input falied";
      return;
    }
    ret = GdcManager::Instance()->MultiOutput(pym_images, gdc_images);
    if (ret) {
      LOGE << "CommonGdcPlugin multi output falied";
      return;
    }
  } else {
    std::shared_ptr<SrcImageFrame> gdc_image;
    ret = GdcManager::Instance()->Input(pym_images[0]);
    if (ret) {
      LOGE << "CommonGdcPlugin input falied";
      return;
    }
    ret = GdcManager::Instance()->Output(pym_images[0], gdc_image);
    if (ret) {
      LOGE << "CommonGdcPlugin output falied";
      return;
    }
    gdc_images.push_back(gdc_image);
  }

  auto gdc_message = std::make_shared<CustomGdcMessage>(
      GdcManager::Instance(), chn_id, gdc_images);
  gdc_message->time_stamp_ = vio_message->time_stamp_;
  gdc_message->sequence_id_ = vio_message->sequence_id_;
  gdc_message->channel_ = vio_message->channel_;
  gdc_message->image_ = vio_message->image_;
  PushMsg(gdc_message);
}

int CommonGdcPlugin::OnGetVioMessage(XProtoMessagePtr msg) {
  if (is_stop_) {
    return -1;
  }
  auto valid_frame = std::static_pointer_cast<VioMessage>(msg);
  if (!valid_frame) {
    return -1;
  }
  gdc_work_thread_.PostTask(
      std::bind(&CommonGdcPlugin::GetGdcResult, this, valid_frame));
  return 0;
}

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
