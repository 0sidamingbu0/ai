/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: xudong.du
 * @Mail: xudong.du@horizon.ai
 * @Date: 2019-08-01 20:38:52
 * @Version: v0.0.1
 * @Brief: smartplugin impl based on xstream.
 * @Last Modified by: xudong.du
 * @Last Modified time: 2021-01-14 05:01:11
 */

#include "smartplugin_multisource.h"

#include <fstream>

#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "websocketplugin/attribute_convert/attribute_convert.h"
#include "xproto/msgtype/include/xproto_msgtype/protobuf/x3.pb.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
#include "xproto_msgtype/vioplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {

using hobot::vision::BBox;
using hobot::vision::Segmentation;
using horizon::vision::xproto::websocketplugin::AttributeConvert;
using xstream::OutputDataPtr;
int dist_calibration_w = 0;
float dist_fit_factor = 0.0;
float dist_fit_impower = 0.0;
bool dist_smooth = true;

std::string FaceBodySmartMessage::Serialize(int ori_w, int ori_h, int dst_w,
                                            int dst_h) {
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

  // add fps to output
  auto static_msg = proto_frame_message.mutable_statistics_msg_();
  auto channel_id_attrs = static_msg->add_attributes_();
  channel_id_attrs->set_type_("channel_id");
  channel_id_attrs->set_value_(channel_id + 1);
  channel_id_attrs->set_value_string_(std::to_string(channel_id + 1));
  // need to change by product
  xstream::BaseDataVector *face_boxes = nullptr;
  xstream::BaseDataVector *face_lmks = nullptr;
  xstream::BaseDataVector *hand_lmks = nullptr;
  xstream::BaseDataVector *lmks = nullptr;
  xstream::BaseDataVector *mask = nullptr;
  xstream::BaseDataVector *features = nullptr;
  auto name_prefix = [](const std::string name) -> std::string {
    auto pos = name.find('_');
    if (pos == std::string::npos) return "";

    return name.substr(0, pos);
  };

  auto name_postfix = [](const std::string name) -> std::string {
    auto pos = name.rfind('_');
    if (pos == std::string::npos) return "";

    return name.substr(pos + 1);
  };

  std::vector<std::shared_ptr<xstream::XStreamData<hobot::vision::BBox>>>
      face_box_list;
  std::vector<std::shared_ptr<xstream::XStreamData<hobot::vision::BBox>>>
      head_box_list;
  std::vector<std::shared_ptr<xstream::XStreamData<hobot::vision::BBox>>>
      body_box_list;
  std::vector<std::shared_ptr<xstream::XStreamData<hobot::vision::BBox>>>
      hand_box_list;
  std::map<target_key, x3::Target *, cmp_key> smart_target;
  for (const auto &output : smart_result_->datas_) {
    LOGD << "output name: " << output->name_;
    auto prefix = name_prefix(output->name_);
    auto postfix = name_postfix(output->name_);
    if (output->name_ == "face_bbox_list" || output->name_ == "head_box" ||
        output->name_ == "body_box" || postfix == "box") {
      face_boxes = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "box type: " << output->name_
           << ", box size: " << face_boxes->datas_.size();
      bool track_id_valid = true;
      if (face_boxes->datas_.size() > 1) {
        track_id_valid = false;
        for (size_t i = 0; i < face_boxes->datas_.size(); ++i) {
          auto face_box = std::static_pointer_cast<
              xstream::XStreamData<hobot::vision::BBox>>(face_boxes->datas_[i]);
          if (face_box->value.id != 0) {
            track_id_valid = true;
            break;
          }
        }
      }
      for (size_t i = 0; i < face_boxes->datas_.size(); ++i) {
        auto face_box =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
                face_boxes->datas_[i]);
        if (!track_id_valid) {
          face_box->value.id = i + 1;
        }
        if (prefix == "hand") {
          face_box->value.category_name = "hand";
        } else {
          face_box->value.category_name = "person";
        }
        LOGD << output->name_ << " id: " << face_box->value.id
             << " x1: " << face_box->value.x1 << " y1: " << face_box->value.y1
             << " x2: " << face_box->value.x2 << " y2: " << face_box->value.y2;
        if (prefix == "face") {
          face_box_list.push_back(face_box);
        } else if (prefix == "head") {
          head_box_list.push_back(face_box);
        } else if (prefix == "body") {
          body_box_list.push_back(face_box);
        } else if (prefix == "hand") {
          hand_box_list.push_back(face_box);
        } else {
          LOGE << "unsupport box name: " << output->name_;
        }
        target_key face_key(face_box->value.category_name, face_box->value.id);
        if (face_box->value.id != -1) {
          if (smart_target.find(face_key) ==
              smart_target.end()) {  // 新track_id
            auto target = smart_msg->add_targets_();
            target->set_track_id_(face_box->value.id);
            if (prefix == "hand") {
              target->set_type_("hand");
            } else {
              target->set_type_("person");
            }
            smart_target[face_key] = target;
          }

          if (prefix == "face" && dist_calibration_w > 0) {
            // cal distance with face box width
            int face_box_width = face_box->value.x2 - face_box->value.x1;
            if (dist_calibration_w != ori_w) {
              // cam input is not 4K, convert width
              face_box_width = face_box_width * dist_calibration_w / ori_w;
            }
            int dist =
                dist_fit_factor * (pow(face_box_width, dist_fit_impower));
            // 四舍五入法平滑
            if (dist_smooth) {
              dist = round(static_cast<float>(dist) / 10.0) * 10;
            }
            auto attrs = smart_target[face_key]->add_attributes_();
            attrs->set_type_("dist");
            attrs->set_value_(dist);
            attrs->set_value_string_(std::to_string(dist));
          }

          auto proto_box = smart_target[face_key]->add_boxes_();
          proto_box->set_type_(prefix);  // "face", "head", "body"
          auto point1 = proto_box->mutable_top_left_();
          point1->set_x_(face_box->value.x1 * x_ratio);
          point1->set_y_(face_box->value.y1 * y_ratio);
          point1->set_score_(face_box->value.score);
          auto point2 = proto_box->mutable_bottom_right_();
          point2->set_x_(face_box->value.x2 * x_ratio);
          point2->set_y_(face_box->value.y2 * y_ratio);
          point2->set_score_(face_box->value.score);

          // body_box在前
          if (prefix == "body" && smart_target[face_key]->boxes__size() > 1) {
            auto body_box_index = smart_target[face_key]->boxes__size() - 1;
            auto body_box =
                smart_target[face_key]->mutable_boxes_(body_box_index);
            auto first_box = smart_target[face_key]->mutable_boxes_(0);
            first_box->Swap(body_box);
          }
        }
      }
    }
    if (output->name_ == "bound_rect_filter") {
      auto bound_rect = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "bound_rect size: " << bound_rect->datas_.size();
      if (bound_rect->datas_.size() == 2) {
        auto bound_box =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
                bound_rect->datas_[0]);
        auto show_enable = std::static_pointer_cast<xstream::XStreamData<bool>>(
            bound_rect->datas_[1]);
        LOGD << output->name_ << " id: " << bound_box->value.id
             << " x1: " << bound_box->value.x1 << " y1: " << bound_box->value.y1
             << " x2: " << bound_box->value.x2 << " y2: " << bound_box->value.y2
             << ", show_enable: " << show_enable->value;
        if (show_enable->value) {
          auto target = smart_msg->add_targets_();
          target->set_track_id_(0);
          target->set_type_("bound");
          auto proto_box = target->add_boxes_();
          proto_box->set_type_("bound_rect");
          auto point1 = proto_box->mutable_top_left_();
          point1->set_x_(bound_box->value.x1 * x_ratio);
          point1->set_y_(bound_box->value.y1 * y_ratio);
          auto point2 = proto_box->mutable_bottom_right_();
          point2->set_x_(bound_box->value.x2 * x_ratio);
          point2->set_y_(bound_box->value.y2 * y_ratio);
        }
      }
    }
    if (output->name_ == "kps" || output->name_ == "lowpassfilter_body_kps") {
      lmks = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "kps size: " << lmks->datas_.size();
      if (lmks->datas_.size() != body_box_list.size()) {
        LOGE << "kps size: " << lmks->datas_.size()
             << ", body_box size: " << body_box_list.size();
      }
      for (size_t i = 0; i < lmks->datas_.size(); ++i) {
        auto lmk = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Landmarks>>(lmks->datas_[i]);
        // 查找对应的track_id
        if (body_box_list[i]->value.id == -1) {
          continue;
        }
        target_key body_key(body_box_list[i]->value.category_name,
                            body_box_list[i]->value.id);
        if (smart_target.find(body_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          auto target = smart_target[body_key];
          auto proto_points = target->add_points_();
          proto_points->set_type_("body_landmarks");
          for (size_t i = 0; i < lmk->value.values.size(); ++i) {
            auto point = proto_points->add_points_();
            point->set_x_(lmk->value.values[i].x * x_ratio);
            point->set_y_(lmk->value.values[i].y * y_ratio);
            point->set_score_(lmk->value.values[i].score);
            LOGD << "x: " << std::round(lmk->value.values[i].x)
                 << " y: " << std::round(lmk->value.values[i].y)
                 << " score: " << lmk->value.values[i].score << "\n";
          }
        }
      }
    }
    if (output->name_ == "mask") {
      mask = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "mask size: " << mask->datas_.size();
      if (mask->datas_.size() != body_box_list.size()) {
        LOGE << "mask size: " << mask->datas_.size()
             << ", body_box size: " << body_box_list.size();
      }
      for (size_t i = 0; i < mask->datas_.size(); ++i) {
        auto one_mask = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Segmentation>>(mask->datas_[i]);
        if (one_mask->state_ != xstream::DataState::VALID) {
          continue;
        }
        // 查找对应的track_id
        if (body_box_list[i]->value.id == -1) {
          continue;
        }
        target_key body_key(body_box_list[i]->value.category_name,
                            body_box_list[i]->value.id);
        if (smart_target.find(body_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
          continue;
        } else {
          auto body_box = body_box_list[i]->value;
          int x1 = body_box.x1;
          int y1 = body_box.y1;
          int x2 = body_box.x2;
          int y2 = body_box.y2;
          auto mask = one_mask->value;
          int h_w = sqrt(mask.values.size());
          cv::Mat mask_mat(h_w, h_w, CV_32F);

          for (int h = 0; h < h_w; ++h) {
            float *ptr = mask_mat.ptr<float>(h);
            for (int w = 0; w < h_w; ++w) {
              *(ptr + w) = (mask.values)[h * h_w + w];
            }
          }
          float max_ratio = 1;
          int width = x2 - x1;
          int height = y2 - y1;

          float w_ratio = static_cast<float>(width) / h_w;
          float h_ratio = static_cast<float>(height) / h_w;
          if (w_ratio >= 4 && h_ratio >= 4) {
            max_ratio = w_ratio < h_ratio ? w_ratio : h_ratio;
            if (max_ratio >= 4) {
              max_ratio = 4;
            }
            width = width / max_ratio;
            height = height / max_ratio;
          }
          cv::resize(mask_mat, mask_mat, cv::Size(width, height));
          cv::Mat mask_gray(height, width, CV_8UC1);
          mask_gray.setTo(0);
          std::vector<std::vector<cv::Point>> contours;

          for (int h = 0; h < height; ++h) {
            uchar *p_gray = mask_gray.ptr<uchar>(h);
            const float *p_mask = mask_mat.ptr<float>(h);
            for (int w = 0; w < width; ++w) {
              if (p_mask[w] > 0) {
                // 这个点在人体内
                p_gray[w] = 1;
              } else {
              }
            }
          }
          mask_mat.release();
          cv::findContours(mask_gray, contours, cv::noArray(), cv::RETR_CCOMP,
                           cv::CHAIN_APPROX_NONE);

          mask_gray.release();
          auto target = smart_target[body_key];
          auto Points = target->add_points_();
          Points->set_type_("mask");
          for (size_t i = 0; i < contours.size(); i++) {
            auto one_line = contours[i];
            for (size_t j = 0; j < one_line.size(); j += 4) {
              auto point = Points->add_points_();
              point->set_x_((contours[i][j].x * max_ratio + x1) * x_ratio);
              point->set_y_((contours[i][j].y * max_ratio + y1) * y_ratio);
              point->set_score_(0);
            }
          }
          contours.clear();
          std::vector<std::vector<cv::Point>>(contours).swap(contours);
        }
      }
    }

    if (output->name_ == "age") {
      auto ages = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "age size: " << ages->datas_.size();
      if (ages->datas_.size() != face_box_list.size()) {
        LOGE << "ages size: " << ages->datas_.size()
             << ", face_box size: " << face_box_list.size();
      }
      for (size_t i = 0; i < ages->datas_.size(); ++i) {
        auto age =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::Age>>(
                ages->datas_[i]);
        // 查找对应的track_id
        if (face_box_list[i]->value.id == -1) {
          continue;
        }
        target_key face_key(face_box_list[i]->value.category_name,
                            face_box_list[i]->value.id);
        if (smart_target.find(face_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          if (age->state_ != xstream::DataState::VALID) {
            LOGE << "-1 -1 -1";
            continue;
          }
          auto target = smart_target[face_key];
          auto attrs = target->add_attributes_();
          attrs->set_type_("age");
          attrs->set_value_((age->value.min + age->value.max) / 2);
          attrs->set_score_(age->value.score);
          attrs->set_value_string_(
              std::to_string((age->value.min + age->value.max) / 2));
          LOGD << " " << age->value.min << " " << age->value.max;
        }
      }
    }

    if (output->name_ == "gender") {
      auto genders = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "gender size: " << genders->datas_.size();
      if (genders->datas_.size() != face_box_list.size()) {
        LOGE << "genders size: " << genders->datas_.size()
             << ", face_box size: " << face_box_list.size();
      }
      for (size_t i = 0; i < genders->datas_.size(); ++i) {
        auto gender = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Gender>>(genders->datas_[i]);
        // 查找对应的track_id
        if (face_box_list[i]->value.id == -1) {
          continue;
        }
        target_key face_key(face_box_list[i]->value.category_name,
                            face_box_list[i]->value.id);
        if (smart_target.find(face_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          if (genders->state_ != xstream::DataState::VALID) {
            LOGE << "-1";
            continue;
          }
          auto target = smart_target[face_key];
          auto attrs = target->add_attributes_();
          attrs->set_type_("gender");
          attrs->set_value_(gender->value.value);
          attrs->set_score_(gender->value.score);
          auto gender_des = AttributeConvert::Instance().GetAttrDes(
              "gender", gender->value.value);
          attrs->set_value_string_(gender_des);
          LOGD << " " << gender->value.value;
        }
      }
    }

    if (output->name_ == "face_mask") {
      auto attributes = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << output->name_ << " size: " << attributes->datas_.size();
      if (attributes->datas_.size() != face_box_list.size()) {
        LOGE << "face mask size: " << attributes->datas_.size()
             << ", face_box size: " << face_box_list.size();
      }
      for (size_t i = 0; i < attributes->datas_.size(); ++i) {
        auto attribute = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Attribute<int>>>(
            attributes->datas_[i]);
        // 查找对应的track_id
        if (face_box_list[i]->value.id == -1) {
          continue;
        }
        target_key face_key(face_box_list[i]->value.category_name,
                            face_box_list[i]->value.id);
        if (smart_target.find(face_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          if (attribute->state_ != xstream::DataState::VALID) {
            LOGE << "-1";
            continue;
          }
          auto target = smart_target[face_key];
          auto attrs = target->add_attributes_();
          attrs->set_type_(output->name_);
          attrs->set_value_(attribute->value.value);
          attrs->set_score_(attribute->value.score);
          LOGD << "value: " << attribute->value.value;
        }
      }
    }
    if (output->name_ == "face_feature") {
      features = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << output->name_ << " size: " << features->datas_.size();

      if (features->datas_.size() != face_box_list.size()) {
        LOGI << "face feature size: " << features->datas_.size()
             << ", face_box size: " << face_box_list.size();
      }
      for (size_t i = 0; i < features->datas_.size(); ++i) {
        if (face_box_list[i]->value.id == -1) {
          continue;
        }
        target_key face_key(face_box_list[i]->value.category_name,
                            face_box_list[i]->value.id);
        // 查找对应的track_id
        if (smart_target.find(face_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          auto one_person_feature =
              std::static_pointer_cast<xstream::BaseDataVector>(
                  features->datas_[i]);
          auto target = smart_target[face_key];
          for (size_t idx = 0; idx < one_person_feature->datas_.size(); idx++) {
            auto one_feature = std::static_pointer_cast<
                xstream::XStreamData<hobot::vision::Feature>>(
                one_person_feature->datas_[idx]);
            if (one_feature->state_ != xstream::DataState::VALID) {
              LOGE << "-1";
              continue;
            }
            auto feature = target->add_float_arrays_();
            feature->set_type_(output->name_);
            for (auto item : one_feature->value.values) {
              feature->add_value_(item);
            }
          }
        }
      }
    }
    if (output->name_ == "hand_lmk" ||
        output->name_ == "lowpassfilter_hand_lmk") {
      hand_lmks = dynamic_cast<xstream::BaseDataVector *>(output.get());
      LOGD << "hand_lmk size: " << hand_lmks->datas_.size();
      if (hand_lmks->datas_.size() != hand_box_list.size()) {
        LOGE << "hand_lmk size: " << hand_lmks->datas_.size()
             << ", hand_box size: " << hand_box_list.size();
      }
      for (size_t i = 0; i < hand_lmks->datas_.size(); ++i) {
        auto lmk = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Landmarks>>(
            hand_lmks->datas_[i]);
        // 查找对应的track_id
        if (hand_box_list[i]->value.id == -1) {
          continue;
        }
        target_key hand_key(hand_box_list[i]->value.category_name,
                            hand_box_list[i]->value.id);
        if (smart_target.find(hand_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          auto target = smart_target[hand_key];
          auto proto_points = target->add_points_();
          proto_points->set_type_("hand_landmarks");
          for (size_t i = 0; i < lmk->value.values.size(); ++i) {
            auto point = proto_points->add_points_();
            point->set_x_(lmk->value.values[i].x * x_ratio);
            point->set_y_(lmk->value.values[i].y * y_ratio);
            point->set_score_(lmk->value.values[i].score);
            LOGD << "x: " << std::round(lmk->value.values[i].x)
                 << " y: " << std::round(lmk->value.values[i].y)
                 << " score: " << lmk->value.values[i].score << "\n";
          }
        }
      }
    }
    if (output->name_ == "lowpassfilter_lmk_106pts") {
      face_lmks = dynamic_cast<xstream::BaseDataVector *>(output.get());
      if (face_lmks->datas_.size() != face_box_list.size()) {
        LOGE << "lmk_106pts size: " << face_lmks->datas_.size()
             << ", hand_box size: " << face_box_list.size();
      }
      for (size_t i = 0; i < face_lmks->datas_.size(); ++i) {
        auto lmk = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Landmarks>>(
            face_lmks->datas_[i]);
        // 查找对应的track_id
        if (face_box_list[i]->value.id == -1) {
          continue;
        }
        target_key lmk106pts_key(face_box_list[i]->value.category_name,
                                 face_box_list[i]->value.id);
        if (smart_target.find(lmk106pts_key) == smart_target.end()) {
          LOGE << "Not found the track_id target";
        } else {
          auto target = smart_target[lmk106pts_key];
          auto proto_points = target->add_points_();
          proto_points->set_type_("lmk_106pts");
          for (size_t i = 0; i < lmk->value.values.size(); ++i) {
            auto point = proto_points->add_points_();
            point->set_x_(lmk->value.values[i].x * x_ratio);
            point->set_y_(lmk->value.values[i].y * y_ratio);
            point->set_score_(lmk->value.values[i].score);
            LOGD << "x: " << std::round(lmk->value.values[i].x)
                 << " y: " << std::round(lmk->value.values[i].y)
                 << " score: " << lmk->value.values[i].score;
          }
        }
      }
    }
  }
  // todo
  proto_frame_message.SerializeToString(&proto_str);
  return proto_str;
}

FaceBodySmartPlugin::FaceBodySmartPlugin(const std::string &config_file)
    : SmartPlugin(config_file) {}

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
