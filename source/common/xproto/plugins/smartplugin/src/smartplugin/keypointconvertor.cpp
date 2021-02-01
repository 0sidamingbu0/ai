/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: xue.liang
 * @Mail: xue.liang@horizon.ai
 * @Date: 2020-12-15 20:38:52
 * @Version: v0.0.1
 * @Brief:
 * @Last Modified by: xue.liang
 * @Last Modified time: 2020-12-16 22:41:30
 */

#include "smartplugin/keypointconvertor.h"

#include "hobotlog/hobotlog.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin {
void KeyPointConvertor::ConverKeyPoint(xstream::OutputDataPtr xstream_out) {
  xstream::BaseDataVector *kps_list = nullptr;
  xstream::BaseDataVector *face_lmks = nullptr;
  xstream::BaseDataVector *face_box_list = nullptr;
  xstream::BaseDataVector *head_box_list = nullptr;
  xstream::BaseDataVector *body_box_list = nullptr;

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

  for (const auto &output : xstream_out->datas_) {
    LOGD << output->name_ << ", type is " << output->type_;
    auto prefix = name_prefix(output->name_);
    auto postfix = name_postfix(output->name_);
    if (output->name_ == "kps" || output->name_ == "lowpassfilter_body_kps") {
      kps_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "body_box" || (prefix == "body" && postfix == "box")) {
      body_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "head_box" || (prefix == "head" && postfix == "box")) {
      head_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "lowpassfilter_lmk_106pts") {
      face_lmks = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "face_bbox_list" ||
        (prefix == "face" && postfix == "box")) {
      face_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
  }
  ConvertFaceLmk(face_lmks);
  ConvertKps(kps_list, body_box_list, head_box_list, face_lmks, face_box_list);
}

void KeyPointConvertor::ConvertFaceLmk(xstream::BaseDataVector *face_lmks) {
  if (!face_lmks) return;

  std::vector<std::shared_ptr<hobot::vision::Point>> tmp_lmk_list;
  for (size_t i = 0; i < face_lmks->datas_.size(); ++i) {
    auto lmk = std::static_pointer_cast<
        xstream::XStreamData<hobot::vision::Landmarks>>(face_lmks->datas_[i]);
    if (lmk->value.values.size() < 106) {
      LOGI << "ConvertFaceLmk recv face lmk less than 106, size:"
           << lmk->value.values.size();
      continue;
    }
#if 0
    auto face_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            face_box_list->datas_[i]);
    if (!face_box) continue;
    // 查找对应的track_id
    if (face_box->value.id == -1) {
      continue;
    }
#endif
    tmp_lmk_list.resize(lmk->value.values.size());
    for (size_t j = 0; j < lmk->value.values.size(); ++j) {
      auto point = std::make_shared<hobot::vision::Point>();
      point->x = lmk->value.values[j].x;
      point->y = lmk->value.values[j].y;
      point->score = lmk->value.values[j].score;
      tmp_lmk_list[j] = point;
      LOGD << "x: " << lmk->value.values[j].x
           << " y: " << lmk->value.values[j].y
           << " score: " << lmk->value.values[j].score;
    }
    // convert
    // 0~37 is the same
    size_t j = 42;
    for (size_t i = 38; i < 43; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    j = 51;
    for (size_t i = 43; i < 47; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    j = 58;
    for (size_t i = 47; i < 52; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    j = 66;
    for (size_t i = 52; i < 54; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    j = 69;
    for (size_t i = 54; i < 57; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    lmk->value.values[57].x = tmp_lmk_list[73]->x;
    lmk->value.values[57].y = tmp_lmk_list[73]->y;
    lmk->value.values[57].score = tmp_lmk_list[73]->score;

    j = 75;
    for (size_t i = 58; i < 60; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    j = 78;
    for (size_t i = 60; i < 63; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j++;
    }

    lmk->value.values[63].x = tmp_lmk_list[82]->x;
    lmk->value.values[63].y = tmp_lmk_list[82]->y;
    lmk->value.values[63].score = tmp_lmk_list[82]->score;

    j = 41;
    for (size_t i = 64; i < 68; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j--;
    }

    j = 50;
    for (size_t i = 68; i < 72; ++i) {
      lmk->value.values[i].x = tmp_lmk_list[j]->x;
      lmk->value.values[i].y = tmp_lmk_list[j]->y;
      lmk->value.values[i].score = tmp_lmk_list[j]->score;
      j--;
    }

    lmk->value.values[72].x = tmp_lmk_list[68]->x;
    lmk->value.values[72].y = tmp_lmk_list[68]->y;
    lmk->value.values[72].score = tmp_lmk_list[68]->score;

    lmk->value.values[73].x = tmp_lmk_list[72]->x;
    lmk->value.values[73].y = tmp_lmk_list[72]->y;
    lmk->value.values[73].score = tmp_lmk_list[72]->score;

    // lmk->value.values[74].x = tmp_lmk_list[74]->x;
    // lmk->value.values[74].y = tmp_lmk_list[74]->y;
    // lmk->value.values[74].score = tmp_lmk_list[74]->score;

    lmk->value.values[75].x = tmp_lmk_list[77]->x;
    lmk->value.values[75].y = tmp_lmk_list[77]->y;
    lmk->value.values[75].score = tmp_lmk_list[77]->score;

    lmk->value.values[76].x = tmp_lmk_list[81]->x;
    lmk->value.values[76].y = tmp_lmk_list[81]->y;
    lmk->value.values[76].score = tmp_lmk_list[81]->score;

    lmk->value.values[77].x = tmp_lmk_list[83]->x;
    lmk->value.values[77].y = tmp_lmk_list[83]->y;
    lmk->value.values[77].score = tmp_lmk_list[83]->score;

    lmk->value.values[78].x = tmp_lmk_list[55]->x;
    lmk->value.values[78].y = tmp_lmk_list[55]->y;
    lmk->value.values[78].score = tmp_lmk_list[55]->score;

    lmk->value.values[79].x = tmp_lmk_list[65]->x;
    lmk->value.values[79].y = tmp_lmk_list[65]->y;
    lmk->value.values[79].score = tmp_lmk_list[65]->score;

    lmk->value.values[80].x = tmp_lmk_list[56]->x;
    lmk->value.values[80].y = tmp_lmk_list[56]->y;
    lmk->value.values[80].score = tmp_lmk_list[56]->score;

    lmk->value.values[81].x = tmp_lmk_list[64]->x;
    lmk->value.values[81].y = tmp_lmk_list[64]->y;
    lmk->value.values[81].score = tmp_lmk_list[64]->score;

    lmk->value.values[82].x = tmp_lmk_list[57]->x;
    lmk->value.values[82].y = tmp_lmk_list[57]->y;
    lmk->value.values[82].score = tmp_lmk_list[57]->score;

    lmk->value.values[83].x = tmp_lmk_list[63]->x;
    lmk->value.values[83].y = tmp_lmk_list[63]->y;
    lmk->value.values[83].score = tmp_lmk_list[63]->score;

    // point 84~105 is the same
    tmp_lmk_list.clear();
  }
}

void KeyPointConvertor::ConvertKps(xstream::BaseDataVector *kps_list,
                                   xstream::BaseDataVector *body_box_list,
                                   xstream::BaseDataVector *head_box_list,
                                   xstream::BaseDataVector *face_lmks,
                                   xstream::BaseDataVector *face_box_list) {
  if (kps_list == nullptr || body_box_list == nullptr ||
      head_box_list == nullptr)
    return;
  std::vector<std::shared_ptr<hobot::vision::Point>> tmp_lmk_list;
  for (size_t i = 0; i < kps_list->datas_.size(); ++i) {
    auto lmk = std::static_pointer_cast<
        xstream::XStreamData<hobot::vision::Landmarks>>(kps_list->datas_[i]);
    if (!lmk) continue;
    if (lmk->value.values.size() < 17) {
      LOGI << "UpdateHandTrackID not body kps, error!!!";
      continue;
    }

    auto body_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            body_box_list->datas_[i]);
    if (!body_box) continue;
    int32_t track_id = body_box->value.id;
    if (track_id == -1) continue;
    tmp_lmk_list.resize(lmk->value.values.size());
    for (size_t j = 0; j < lmk->value.values.size(); ++j) {
      auto point = std::make_shared<hobot::vision::Point>();
      point->x = lmk->value.values[j].x;
      point->y = lmk->value.values[j].y;
      point->score = lmk->value.values[j].score;
      tmp_lmk_list[j] = point;
      LOGD << "x: " << lmk->value.values[j].x
           << " y: " << lmk->value.values[j].y
           << " score: " << lmk->value.values[j].score;
    }

    // find box by trackid
    for (size_t k = 0; k < head_box_list->datas_.size(); ++k) {
      auto head_box =
          std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
              head_box_list->datas_[k]);
      if (head_box->value.id != track_id) continue;

      hobot::vision::Point head;
      head.x =
          head_box->value.x1 + (head_box->value.x2 - head_box->value.x1) / 2;
      head.y = head_box->value.y1;
      head.score = head_box->value.score;
      lmk->value.values[0].x = head.x;
      lmk->value.values[0].y = head.y;
      lmk->value.values[0].score = head.score;
      if (!face_lmks || !face_box_list) {
        LOGI << "ConvertToTCLKps recv null face info, get head line to convert";
        hobot::vision::Point jaw;
        jaw.x = head.x;
        jaw.y = head_box->value.y2;
        jaw.score = head_box->value.score;
        lmk->value.values[1].x = jaw.x;
        lmk->value.values[1].y = jaw.y;
        lmk->value.values[1].score = jaw.score;
      }
      break;
    }

    if (face_lmks && face_box_list) {
      for (size_t k = 0; k < face_lmks->datas_.size(); ++k) {
        auto face_lmk = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Landmarks>>(
            face_lmks->datas_[k]);
        auto face_box =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
                face_box_list->datas_[k]);
        if (face_box->value.id != track_id) continue;

        lmk->value.values[1].x = face_lmk->value.values[17].x;
        lmk->value.values[1].y = face_lmk->value.values[17].y;
        lmk->value.values[1].score = face_lmk->value.values[17].score;
        break;
      }
    }
    // convert
    for (size_t j = 2; j < 14; ++j) {
      if (j % 2) {
        lmk->value.values[j].x = tmp_lmk_list[j + 2]->x;
        lmk->value.values[j].y = tmp_lmk_list[j + 2]->y;
        lmk->value.values[j].score = tmp_lmk_list[j + 2]->score;
      } else {
        lmk->value.values[j].x = tmp_lmk_list[j + 4]->x;
        lmk->value.values[j].y = tmp_lmk_list[j + 4]->y;
        lmk->value.values[j].score = tmp_lmk_list[j + 4]->score;
      }
    }

    lmk->value.values.resize(14);
    tmp_lmk_list.clear();
  }
}

}  // namespace smartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
