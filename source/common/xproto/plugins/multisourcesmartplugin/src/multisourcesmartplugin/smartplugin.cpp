/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-30 22:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include "multisourcesmartplugin/smartplugin.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "hobotlog/hobotlog.hpp"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto/plugin/xpluginasync.h"

#include "hobotxsdk/xstream_sdk.h"
#include "hobotxstream/json_key.h"
#include "horizon/vision/util.h"
#include "horizon/vision_type/vision_error.h"
#include "horizon/vision_type/vision_type.hpp"
#include "multisourcesmartplugin/convert.h"
#include "multisourcesmartplugin/runtime_monitor.h"
#include "multisourcesmartplugin/smart_config.h"

#include "xproto_msgtype/vioplugin_data.h"
#include "xproto_msgtype/gdcplugin_data.h"
#include "xproto_msgtype/commongdcplugin_data.h"
#include "xproto_msgtype/protobuf/x3.pb.h"

#include "multisourcesmartplugin/data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;

using horizon::vision::xproto::basic_msgtype::VioMessage;
using ImageFramePtr = std::shared_ptr<hobot::vision::ImageFrame>;
using XStreamImageFramePtr = xstream::XStreamData<ImageFramePtr>;
using horizon::vision::xproto::basic_msgtype::CustomGdcMessage;
using horizon::vision::xproto::basic_msgtype::IpmImageMessage;

using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;
using horizon::vision::xproto::vioplugin::MultiVioMessage;
using horizon::vision::xproto::multisourcedata::MultiSourceSmartResult;

XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_SMART_MESSAGE)

std::string CustomSmartMessage::Serialize() {
  for (const auto &output : smart_result_->datas_) {
    LOGI << "output name: " << output->name_;
    if (output->name_ == "face_bbox_list") {
      auto face_boxes = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGI << "box type: " << output->name_
           << ", box size: " << face_boxes->datas_.size();
      for (size_t i = 0; i < face_boxes->datas_.size(); ++i) {
        auto face_box =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
                face_boxes->datas_[i]);
        LOGI << output->name_ << " id: " << face_box->value.id
             << " x1: " << face_box->value.x1 << " y1: " << face_box->value.y1
             << " x2: " << face_box->value.x2 << " y2: " << face_box->value.y2;
      }
    }
  }
  return "";
}

std::string CustomSmartMessage::Serialize(int ori_w, int ori_h, int dst_w,
                                          int dst_h) {
  if (ori_w < 0 || ori_h < 0 || dst_w < 0 || dst_h < 0) {
    LOGF << "param error";
    return "";
  }
  float x_ratio = 1.0 * dst_w / ori_w;
  float y_ratio = 1.0 * dst_h / ori_h;
  std::string proto_str;
  x3::FrameMessage proto_frame_message;
  proto_frame_message.set_timestamp_(time_stamp);
  auto smart_msg = proto_frame_message.mutable_smart_msg_();
  smart_msg->set_timestamp_(time_stamp);
  smart_msg->set_error_code_(0);

  // add fps to output
  auto static_msg = proto_frame_message.mutable_statistics_msg_();
  auto channel_id_attrs = static_msg->add_attributes_();
  channel_id_attrs->set_type_("channel_id");
  channel_id_attrs->set_value_(channel_id + 1);
  channel_id_attrs->set_value_string_(std::to_string(channel_id + 1));
  // need to change by product
  for (const auto &output : smart_result_->datas_) {
    if (output->name_ == "cycle_box" || output->name_ == "person_box" ||
        output->name_ == "vehicle_box") {
      xstream::BaseDataVector *box_list =
          dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "box size:" << box_list->datas_.size();
      for (size_t i = 0; i < box_list->datas_.size(); ++i) {
        auto one_box =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
                box_list->datas_[i]);
        if (one_box->state_ == xstream::DataState::VALID) {
          auto target = smart_msg->add_targets_();
          auto proto_box = target->add_boxes_();
          if (output->name_ == "cycle_box") {
            proto_box->set_type_("cyclist");
          } else if (output->name_ == "person_box") {
            proto_box->set_type_("persion");
          } else if (output->name_ == "vehicle_box") {
            proto_box->set_type_("vehicle");
          }
          auto point1 = proto_box->mutable_top_left_();
          point1->set_x_(one_box->value.x1 * x_ratio);
          point1->set_y_(one_box->value.y1 * y_ratio);
          point1->set_score_(one_box->value.score);
          auto point2 = proto_box->mutable_bottom_right_();
          point2->set_x_(one_box->value.x2 * x_ratio);
          point2->set_y_(one_box->value.y2 * y_ratio);
          point2->set_score_(one_box->value.score);
        }
      }
    }
    if (output->name_ == "mask") {
      auto masks = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "mask type: " << output->name_
           << ", masks size: " << masks->datas_.size();
      for (size_t i = 0; i < masks->datas_.size(); ++i) {
        auto one_mask = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Segmentation>>(
            masks->datas_[i]);
        auto seg_width = one_mask->value.width;
        auto seg_height = one_mask->value.height;
        if (one_mask->value.values.size() !=
            static_cast<size_t>(seg_width * seg_height)) {
          LOGE << "mask data occur errors, " << seg_width << ", " << seg_height;
          break;
        }
        auto target = smart_msg->add_targets_();
        auto float_matrix = target->add_float_matrixs_();
        float_matrix->set_type_("parking_mask");
        int mask_value_idx = 0;
        for (int h = 0; h < seg_height; h += 5) {
          auto float_array = float_matrix->add_arrays_();
          for (int w = 0; w < seg_width; w += 5) {
            mask_value_idx = h * seg_width + w;
            float_array->add_value_(one_mask->value.values[mask_value_idx]);
          }
        }
      }
    }
    if (output->name_ == "parking_box") {
      xstream::BaseDataVector *box_list =
          dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "parking size:" << box_list->datas_.size();
      for (size_t i = 0; i < box_list->datas_.size(); ++i) {
        auto one_box = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Oriented_BBox>>(
            box_list->datas_[i]);
        if (one_box->state_ == xstream::DataState::VALID) {
          auto target = smart_msg->add_targets_();
          auto proto_points = target->add_points_();
          proto_points->set_type_("parking");
          auto point_1 = proto_points->add_points_();
          point_1->set_x_(one_box->value.x1 * x_ratio);
          point_1->set_y_(one_box->value.y1 * y_ratio);
          point_1->set_score_(one_box->value.score);
          auto point_2 = proto_points->add_points_();
          point_2->set_x_(one_box->value.x2 * x_ratio);
          point_2->set_y_(one_box->value.y2 * y_ratio);
          point_2->set_score_(one_box->value.score);
          auto point_3 = proto_points->add_points_();
          point_3->set_x_(one_box->value.x3 * x_ratio);
          point_3->set_y_(one_box->value.y3 * y_ratio);
          point_3->set_score_(one_box->value.score);
          auto point_4 = proto_points->add_points_();
          point_4->set_x_(one_box->value.x4 * x_ratio);
          point_4->set_y_(one_box->value.y4 * y_ratio);
          point_4->set_score_(one_box->value.score);
        }
      }
    }
    if (output->name_ == "corner") {
      xstream::BaseDataVector *corner_list =
          dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "corner size:" << corner_list->datas_.size();
      if (corner_list->datas_.size() > 0) {
        auto target = smart_msg->add_targets_();
        auto proto_points = target->add_points_();
        proto_points->set_type_("corner");
        for (size_t i = 0; i < corner_list->datas_.size(); ++i) {
          auto one_corner = std::static_pointer_cast<
              xstream::XStreamData<hobot::vision::Point>>(
              corner_list->datas_[i]);
          if (one_corner->state_ == xstream::DataState::VALID) {
            auto point = proto_points->add_points_();
            point->set_x_(one_corner->value.x * x_ratio);
            point->set_y_(one_corner->value.y * y_ratio);
            point->set_score_(one_corner->value.score);
          }
        }
      }
    }
  }
  // todo
  proto_frame_message.SerializeToString(&proto_str);
  return proto_str;
}

void *CustomSmartMessage::ConvertData() {
  MultiSourceSmartResult *multi_smart_ret = new MultiSourceSmartResult();
  for (const auto &output : smart_result_->datas_) {
    LOGI << "output name: " << output->name_;
    if (output->name_ == "cycle_box" || output->name_ == "person_box" ||
        output->name_ == "vehicle_box" || output->name_ == "rear_box" ||
        output->name_ == "parkinglock_box") {
      auto boxes = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGI << "box type: " << output->name_
           << ", boxes size: " << boxes->datas_.size();
      for (size_t i = 0; i < boxes->datas_.size(); ++i) {
        auto box =
          std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            boxes->datas_[i]);
        hobot::vision::BBox tmp_box(box->value.x1, box->value.y1,
                                    box->value.x2, box->value.y2,
                                    box->value.score, box->value.id,
                                    output->name_);
        multi_smart_ret->bbox_list_[channel_id].push_back(tmp_box);
      }
    }
    if (output->name_ == "drivingarea_mask") {
      auto masks = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGI << "mask type: " << output->name_
           << ", masks size: " << masks->datas_.size();
      for (size_t i = 0; i < masks->datas_.size(); ++i) {
        auto mask =
          std::static_pointer_cast<xstream::XStreamData<
            hobot::vision::Segmentation>>(masks->datas_[i]);
        hobot::vision::Segmentation tmp_seg;
        tmp_seg.width = mask->value.width;
        tmp_seg.height = mask->value.height;
        tmp_seg.score = mask->value.score;
        for (const auto item : mask->value.values) {
          tmp_seg.values.push_back(item);
        }
        for (const auto item : mask->value.pixel_score) {
          tmp_seg.pixel_score.push_back(item);
        }
        multi_smart_ret->segmentation_list_[channel_id].push_back(tmp_seg);
      }
    }
  }
  return multi_smart_ret;
}

SmartPlugin::SmartPlugin(const std::string &config_file) {
  config_file_ = config_file;
  LOGI << "smart config file:" << config_file_;
  Json::Value cfg_jv;
  std::ifstream infile(config_file_);
  infile >> cfg_jv;
  config_.reset(new JsonConfigWrapper(cfg_jv));
}

// 覆盖workflow的source id配置
int SmartPlugin::OverWriteSourceNum(const std::string &cfg_file,
                                    int source_num) {
  std::fstream cfg_file_stream(cfg_file, std::fstream::out | std::fstream::in);
  LOGI << "overwrite workflow cfg_file: " << cfg_file;
  if (!cfg_file_stream.good()) {
    LOGE << "Open failed:" << cfg_file;
    return -1;
  }
  Json::Value config;
  cfg_file_stream >> config;
  if (source_num <= 0) {
    config.removeMember(xstream::kSourceNum);
  } else {
    config[xstream::kSourceNum] = source_num;
  }
  cfg_file_stream.seekp(0, std::ios_base::beg);
  cfg_file_stream << config;
  cfg_file_stream.close();
  return 0;
}

int SmartPlugin::Init() {
  auto workflows_cfg = config_->GetSubConfig("workflows");

  // 初始化相关字段
  int workflow_cnt = workflows_cfg->ItemCount();
  LOGI << "workflow count: " << workflow_cnt;

  sdk_.resize(workflow_cnt);
  source_map_.resize(workflow_cnt);

  monitor_.reset(new RuntimeMonitor());

  for (int i = 0; i < workflow_cnt; i++) {
    auto item_cfg = workflows_cfg->GetSubConfig(i);
    auto xstream_workflow_file =
        item_cfg->GetSTDStringValue("xstream_workflow_file");
    auto source_list = item_cfg->GetIntArray("source_list");

    sdk_[i].reset(xstream::XStreamSDK::CreateSDK());

    if (item_cfg->GetBoolValue("overwrite")) {
      OverWriteSourceNum(xstream_workflow_file, source_list.size());
    }

    sdk_[i]->SetConfig("config_file", xstream_workflow_file);

    if (item_cfg->GetBoolValue("enable_profile")) {
      sdk_[i]->SetConfig("profiler", "on");
      sdk_[i]->SetConfig("profiler_file",
                         item_cfg->GetSTDStringValue("profile_log_path"));
    }

    if (sdk_[i]->Init() != 0) {
      return kHorizonVisionInitFail;
    }

    sdk_[i]->SetCallback(
        std::bind(&SmartPlugin::OnCallback, this, std::placeholders::_1));

    // 建立source id到workflow的映射关系
    for (unsigned int j = 0; j < source_list.size(); j++) {
      auto &tmp = source_target_[source_list[j]];
      auto iter = std::find(tmp.begin(), tmp.end(), i);
      if (iter == tmp.end()) {
        tmp.push_back(i);
      }
      // 加入到对应xstream instance的source list中
      source_map_[i].push_back(source_list[j]);
    }
  }

  RegisterMsg(TYPE_IMAGE_MESSAGE,
              std::bind(&SmartPlugin::Feed, this, std::placeholders::_1));

  RegisterMsg(TYPE_GDC_MESSAGE,
              std::bind(&SmartPlugin::FeedIpm, this, std::placeholders::_1));

  RegisterMsg(TYPE_MULTI_IMAGE_MESSAGE,
              std::bind(&SmartPlugin::FeedMulti, this, std::placeholders::_1));

  return XPluginAsync::Init();
}

// feed video frame to xstreamsdk.
int SmartPlugin::Feed(XProtoMessagePtr msg) {
  // 1. parse valid frame from msg
  auto valid_frame = std::static_pointer_cast<VioMessage>(msg);
  auto image_num = valid_frame->image_.size();
  for (uint32_t image_idx = 0; image_idx < image_num; image_idx++) {
    int source_id = valid_frame->image_[image_idx]->channel_id;
    int frame_id = valid_frame->image_[image_idx]->frame_id;
    auto iter = source_target_.find(source_id);
    if (iter == source_target_.end()) {
      LOGF << "Unknow Source ID: " << source_id;
      return 0;
    }
    LOGD << "Source ID: " << source_id << ", frame_id = " << frame_id;

    auto target_sdk = iter->second;
    for (unsigned int i = 0; i < target_sdk.size(); i++) {
      int sdk_idx = target_sdk[i];
      // 创建xstream的输入
      xstream::InputDataPtr input = Convertor::ConvertInput(
          valid_frame.get(), image_idx, GetWorkflowInputImageName(sdk_idx));
      // 设置Source ID
      for (unsigned int j = 0; j < source_map_[sdk_idx].size(); j++) {
        if (source_map_[sdk_idx][j] == source_id) {
          input->source_id_ = j;
          break;
        }
      }

      input->context_ = (const void *)((uintptr_t)sdk_idx);  // NOLINT

      SmartInput *input_wrapper = new SmartInput();
      input_wrapper->frame_info = valid_frame;
      input_wrapper->context = input_wrapper;
      monitor_->PushFrame(input_wrapper, image_idx);

      if (sdk_[sdk_idx]->AsyncPredict(input) <= 0) {
        LOGW << "Async failed: " << sdk_idx << ", source id: " << source_id;

        auto input = monitor_->PopFrame(source_id, frame_id);
        if (input.ref_count == 0) {
          delete static_cast<SmartInput *>(input.context);
        }

        continue;
      }
    }
  }
  return 0;
}

int SmartPlugin::FeedMulti(XProtoMessagePtr msg) {
  auto multi_frame = std::static_pointer_cast<MultiVioMessage>(msg);

  for (size_t m = 0; m < multi_frame->multi_vio_img_.size(); m++) {
    auto valid_frame = multi_frame->multi_vio_img_[m];
    int source_id = valid_frame->image_[0]->channel_id;
    int frame_id = valid_frame->image_[0]->frame_id;

    auto iter = source_target_.find(source_id);
    if (iter == source_target_.end()) {
      LOGF << "Unknow Source ID: " << source_id;
      return 0;
    }
    LOGD << "Source ID: " << source_id;

    auto target_sdk = iter->second;
    for (unsigned int i = 0; i < target_sdk.size(); i++) {
      int sdk_idx = target_sdk[i];
      // 创建xstream的输入
      xstream::InputDataPtr input = Convertor::ConvertInput(
          valid_frame.get(), GetWorkflowInputImageName(sdk_idx));
      // 设置Source ID
      for (unsigned int j = 0; j < source_map_[sdk_idx].size(); j++) {
        if (source_map_[sdk_idx][j] == source_id) {
          input->source_id_ = j;
          break;
        }
      }

      input->context_ = (const void *)((uintptr_t)sdk_idx);  // NOLINT

      SmartInput *input_wrapper = new SmartInput();
      input_wrapper->frame_info = valid_frame;
      input_wrapper->context = input_wrapper;
      monitor_->PushFrame(input_wrapper);

      if (sdk_[sdk_idx]->AsyncPredict(input) < 0) {
        LOGW << "Async failed: " << sdk_idx << ", source id: " << source_id;

        auto input = monitor_->PopFrame(source_id, frame_id);
        if (input.ref_count == 0) {
          delete static_cast<SmartInput *>(input.context);
        }
        continue;
      }
    }
  }
  return 0;
}

int SmartPlugin::FeedIpm(XProtoMessagePtr msg) {
  auto valid_frame = std::static_pointer_cast<CustomGdcMessage>(msg);
  LOGI << "FeedIpm(), image num: " << valid_frame->image_.size();
  return 0;
}

int SmartPlugin::Start() {
  LOGW << "SmartPlugin Start";
  return 0;
}

int SmartPlugin::Stop() {
  LOGW << "SmartPlugin Stop";
  return 0;
}

void SmartPlugin::OnCallback(xstream::OutputDataPtr xstream_out) {
  // On xstream async-predict returned,
  // transform xstream standard output to smart message.
  HOBOT_CHECK(!xstream_out->datas_.empty()) << "Empty XStream Output";

  using xstream::BaseDataVector;
  LOGI << "============Output Call Back============";
  LOGI << "—seq: " << xstream_out->sequence_id_;
  LOGI << "—output_type: " << xstream_out->output_type_;
  LOGI << "—error_code: " << xstream_out->error_code_;
  LOGI << "—error_detail_: " << xstream_out->error_detail_;
  LOGI << "—datas_ size: " << xstream_out->datas_.size();
  for (auto data : xstream_out->datas_) {
    LOGI << "——output data " << data->name_ << " state:"
         << static_cast<std::underlying_type<xstream::DataState>::type>(
                data->state_);

    if (data->error_code_ < 0) {
      LOGI << "——data error: " << data->error_code_;
      continue;
    }
    LOGI << "——data type:" << data->type_ << " name:" << data->name_;
  }
  LOGI << "============Output Call Back End============";

  int sdk_idx = (int)((uintptr_t)xstream_out->context_);  // NOLINT
  int source_id = source_map_[sdk_idx][xstream_out->source_id_];

  XStreamImageFramePtr *rgb_image = nullptr;

  for (const auto &output : xstream_out->datas_) {
    LOGD << output->name_ << ", type is " << output->type_;
    if (output->name_ == GetWorkflowInputImageName(sdk_idx)) {
      rgb_image = dynamic_cast<XStreamImageFramePtr *>(output.get());
    }
  }
  HOBOT_CHECK(rgb_image);

  auto smart_msg = CreateSmartMessage(xstream_out);
  // Set origin input named "image" as output always.
  smart_msg->time_stamp = rgb_image->value->time_stamp;
  smart_msg->frame_id = rgb_image->value->frame_id;
  smart_msg->channel_id = source_id;
  PushMsg(smart_msg);

  auto input = monitor_->PopFrame(source_id, smart_msg->frame_id);
  if (input.ref_count == 0) {
    delete static_cast<SmartInput *>(input.context);
  }
  monitor_->FrameStatistic();
  smart_msg->frame_fps = monitor_->GetFrameFps();
  LOGI << "smartplugin got result, channel_id: " << source_id
       << ", frame_id: " << smart_msg->frame_id;
}

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
