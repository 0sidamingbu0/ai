/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file ExampleSmartplugin.cpp
 * @brief
 * @author ronghui.zhang
 * @email ronghui.zhang@horizon.ai
 *
 *
 * */
#include <fstream>
#include "ExampleSmartPlugin.h"
#include "xproto_msgtype/protobuf/x3.pb.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace ExampleSmartPlugin {

using xstream::OutputDataPtr;
using horizon::vision::xproto::smartplugin::RuntimeMonitor;
using horizon::vision::xproto::smartplugin::JsonConfigWrapper;

std::string ExampleCustomSmartMessage::Serialize(
  int ori_w, int ori_h, int dst_w, int dst_h) {
  HOBOT_CHECK(ori_w > 0 && ori_h > 0 && dst_w > 0 && dst_h > 0)
      << "Serialize param error";
  float x_ratio = 1.0 * dst_w / ori_w;
  float y_ratio = 1.0 * dst_h / ori_h;
  std::string proto_str;
  x3::FrameMessage proto_frame_message;
  proto_frame_message.set_timestamp_(time_stamp);
  auto smart_msg = proto_frame_message.mutable_smart_msg_();
  smart_msg->set_timestamp_(time_stamp);
  smart_msg->set_error_code_(0);

  std::vector<std::shared_ptr<
      xstream::XStreamData<hobot::vision::BBox>>> detected_box_list;

  for (const auto &output : smart_result->datas_) {
    LOGD << "output name: " << output->name_;

    // output name must be same as workflow global output
    if (output->name_ == "detect_box") {
        auto box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
        LOGD << "box type: " << output->name_
           << ", box size: " << box_list->datas_.size();
        for (size_t i = 0; i < box_list->datas_.size(); i++) {
          auto detect_box = std::static_pointer_cast<
              xstream::XStreamData<hobot::vision::BBox>>(box_list->datas_[i]);
          detected_box_list.push_back(detect_box);

          auto target = smart_msg->add_targets_();
          target->set_type_(detect_box->value.category_name);
          target->set_track_id_(detect_box->value.id);
          auto proto_box = target->add_boxes_();
          auto point1 = proto_box->mutable_top_left_();
          point1->set_x_(detect_box->value.x1 * x_ratio);
          point1->set_y_(detect_box->value.y1 * y_ratio);
          point1->set_score_(detect_box->value.score);
          auto point2 = proto_box->mutable_bottom_right_();
          point2->set_x_(detect_box->value.x2 * x_ratio);
          point2->set_y_(detect_box->value.y2 * y_ratio);
          point2->set_score_(detect_box->value.score);
        }
    } else if (output->name_ == "classify") {
       auto box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
       if (box_list->datas_.size() != detected_box_list.size()) {
         LOGE << "classify box size not equal detect box size";
       }
      for (size_t i = 0; i < box_list->datas_.size(); i++) {
        auto classify_box = std::static_pointer_cast<
              xstream::XStreamData<hobot::vision::BBox>>(box_list->datas_[i]);
        auto target = smart_msg->mutable_targets_(i);
        auto attrs = target->add_attributes_();
        attrs->set_type_("label");
        attrs->set_value_string_(classify_box->value.category_name);
        attrs->set_score_(classify_box->value.score);
      }
    }
  }

  proto_frame_message.SerializeToString(&proto_str);
  return proto_str;
}

ExampleSmartPlugin::ExampleSmartPlugin(const std::string& config_file) :
  SmartPlugin(config_file) {
}

}  // namespace ExampleSmartPlugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
