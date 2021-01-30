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

#include "roiplugin/roiplugin.h"

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "roiplugin/roiprocessor.h"
#include "smartplugin/smartplugin.h"
#include "utils/convert.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace roiplugin {

XPLUGIN_REGISTER_MSG_TYPE(TYPE_ROI_SMART_MESSAGE)

using horizon::vision::xproto::smartplugin::CustomSmartMessage;
using ImageFramePtr = std::shared_ptr<hobot::vision::ImageFrame>;
using XStreamImageFramePtr = xstream::XStreamData<ImageFramePtr>;

RoiPlugin::RoiPlugin(const std::string &config_file) {
  config_file_ = config_file;
}

int RoiPlugin::ParseConfig() {
  std::ifstream ifs(config_file_);
  if (!ifs.is_open()) {
    LOGE << "Open config file " << config_file_ << " failed";
    return kHorizonVisionFailure;
  }
  ifs >> config_;
  ifs.close();

  auto value_js = config_["enable_auto_start"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: enable_auto_start";
  }
  auto_start_ = value_js.asBool();
  value_js = config_["enable_vot"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: enable_vot";
  }
  enable_vot_ = value_js.asBool();
  LOGI << "Roiplugin Parse Config Done, auto start flag:" << auto_start_
       << " enable vot:" << enable_vot_;

  value_js = config_["enable_intelligent_tracking"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: enable_intelligent_tracking";
  }
  enable_intelligent_tracking_ = value_js.asBool();

  RoiCalcConfig config;

  value_js = config_["view_roi_update_thr"];
  config.view_roi_update_thr = value_js.asFloat();
  value_js = config_["view_expansion_coefficient"];
  config.view_expansion_coefficient = value_js.asFloat();
  value_js = config_["view_transition_coefficient"];
  config.view_transition_coefficient = value_js.asInt();
  value_js = config_["track_roi_update_thr"];
  config.track_roi_update_thr = value_js.asFloat();
  value_js = config_["track_expansion_coefficient"];
  config.track_expansion_coefficient = value_js.asFloat();
  value_js = config_["track_roi_display"];
  config.track_roi_display = value_js.asInt();
  value_js = config_["keep_ratio"];
  config.keep_ratio = value_js.asBool();

  RoiProcessor::Instance()->SetConfig(config);
  return kHorizonVisionSuccess;
}

int RoiPlugin::Init() {
  if (is_init_) return kHorizonVisionSuccess;
  LOGI << "RoiPlugin INIT";
  ParseConfig();
  if (!auto_start_) {
    LOGW << "roiplugin no need to start";
    return kHorizonVisionSuccess;
  }

  int nRet =
      RoiProcessor::Instance()->Init(enable_vot_, enable_intelligent_tracking_);
  if (nRet != 0) {
    LOGE << "Roiplugin init precessor fail, return:" << nRet;
    return nRet;
  }

  RegisterMsg(TYPE_SMART_MESSAGE, std::bind(&RoiPlugin::OnGetSmarterResult,
                                            this, std::placeholders::_1));
#if 0
  RegisterMsg(TYPE_DROP_IMAGE_MESSAGE, std::bind(&RoiPlugin::OnGetVioResult,
                                                 this, std::placeholders::_1));
#endif
  is_init_ = true;
  return XPluginAsync::Init();
}

int RoiPlugin::DeInit() {
  if (!is_init_) return kHorizonVisionSuccess;
  return RoiProcessor::Instance()->DeInit();
}

int RoiPlugin::Start() {
  if (!auto_start_) {
    LOGW << "roiplugin no need to start";
    return kHorizonVisionSuccess;
  }
  if (is_running_) return kHorizonVisionSuccess;
  if (!is_init_) return kHorizonVisionFailure;
  RoiProcessor::Instance()->Start();
  is_running_ = true;
  LOGI << "roiplugin start";
  return kHorizonVisionSuccess;
}

int RoiPlugin::Stop() {
  if (!is_running_) return kHorizonVisionSuccess;
  LOGI << "RoiPlugin stop...";
  if (auto_start_ || enable_vot_) RoiProcessor::Instance()->Stop();
  LOGI << "roiplugin stop end";
  return kHorizonVisionSuccess;
}

int RoiPlugin::StartPlugin(uint64_t msg_id) {
  // todo, send plugin init msg if someone need
  LOGI << "roi StartPlugin done msg_id:" << msg_id;
  return kHorizonVisionSuccess;
}

int RoiPlugin::StopPlugin(uint64_t msg_id) {
  // todo, send plugin stop msg if someone need
  LOGI << "roi StopPlugin done";
  return kHorizonVisionSuccess;
}

int RoiPlugin::OnGetSmarterResult(const XProtoMessagePtr &msg) {
  if (!is_running_ || !auto_start_ || !enable_vot_) {
    return kHorizonVisionSuccess;
  }

  switch ((SmartType)smart_type_) {
    case SmartType::SMART_FACE:
    case SmartType::SMART_BODY: {
      auto smart_msg = dynamic_cast<CustomSmartMessage *>(msg.get());
      if (smart_msg) {
        ConvertSmartToRoiData(smart_msg->GetSmartResult(), smart_msg->frame_id,
                              smart_msg->time_stamp);
      }
      break;
    }
    default:
      LOGE << "not support smart_type";
      return kHorizonVisionFailure;
  }
  // todo get result and pushmsg
  return kHorizonVisionSuccess;
}

void ConverVideoData(XStreamImageFramePtr *rgb_image, const int layer,
                     std::shared_ptr<VideoRoiData> video_data) {
  uint32_t width_tmp =
      ((hobot::vision::PymImageFrame *)(rgb_image->value.get()))
          ->down_scale[layer]
          .width;
  uint32_t height_tmp =
      ((hobot::vision::PymImageFrame *)(rgb_image->value.get()))
          ->down_scale[layer]
          .height;
  char *y_addr = reinterpret_cast<char *>(
      ((hobot::vision::PymImageFrame *)(rgb_image->value.get()))
          ->down_scale[layer]
          .y_vaddr);
  char *uv_addr = reinterpret_cast<char *>(
      ((hobot::vision::PymImageFrame *)(rgb_image->value.get()))
          ->down_scale[layer]
          .c_vaddr);
  video_data->width = width_tmp;
  video_data->height = height_tmp;
  video_data->y_virtual_addr =
      static_cast<char *>(malloc(width_tmp * height_tmp));
  video_data->uv_virtual_addr =
      static_cast<char *>(malloc(width_tmp * height_tmp / 2));
  memcpy(video_data->y_virtual_addr, y_addr, width_tmp * height_tmp);
  memcpy(video_data->uv_virtual_addr, uv_addr, width_tmp * height_tmp / 2);
}

void RoiPlugin::ConvertSmartToRoiData(xstream::OutputDataPtr xstream_out,
                                      const uint64_t frame_id,
                                      const uint64_t pts) {
  HOBOT_CHECK(!xstream_out->datas_.empty()) << "Empty XStream Output";
  XStreamImageFramePtr *rgb_image = nullptr;
  for (const auto &output : xstream_out->datas_) {
    LOGD << output->name_ << ", type is " << output->type_;
    if (output->name_ == "rgb_image" || output->name_ == "image") {
      rgb_image = dynamic_cast<XStreamImageFramePtr *>(output.get());
    }
  }
  HOBOT_CHECK(rgb_image);
  int layer = 0;
  std::shared_ptr<VideoRoiData> video_data = std::make_shared<VideoRoiData>();
  video_data->frame_id = frame_id;
  video_data->timestamp = pts;
  video_data->channel = 1;
  HorizonVisionAllocSmartFrame(&video_data->smart_frame);
  Convertor::ConvertOutputToSmartFrame(xstream_out, video_data->smart_frame);
  ConverVideoData(rgb_image, layer, video_data);
  RoiProcessor::Instance()->Input(video_data);

  std::shared_ptr<VideoRoiData> roi_data = std::make_shared<VideoRoiData>();
  roi_data->frame_id = frame_id;
  roi_data->timestamp = pts;
  roi_data->channel = 0;
  layer = 4;
  ConverVideoData(rgb_image, layer, roi_data);
  RoiProcessor::Instance()->Input(roi_data);
}

}  // namespace roiplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
