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

#include "smartplugin/mergehandbody.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

#include "hobotlog/hobotlog.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin {
void MergeHandBody::UpdateHandTrackID(xstream::OutputDataPtr xstream_out) {
  xstream::BaseDataVector *kps_list = nullptr;
  xstream::BaseDataVector *hand_lmk_list = nullptr;
  xstream::BaseDataVector *body_box_list = nullptr;
  xstream::BaseDataVector *hand_box_list = nullptr;
  xstream::BaseDataVector *gesture_list = nullptr;
  xstream::BaseDataVector *face_boxe_list = nullptr;
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

    if (output->name_ == "hand_lmk" ||
        output->name_ == "lowpassfilter_hand_lmk") {
      hand_lmk_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
    if (output->name_ == "body_box" || prefix == "body") {
      body_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
    if (prefix == "hand" && postfix == "box") {
      hand_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "gesture_vote") {
      gesture_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
    if (output->name_ == "face_bbox_list") {
      face_boxe_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
  }

  UpdateHandTrackID(body_box_list, kps_list, hand_box_list, hand_lmk_list,
                    face_boxe_list);
  FilterGesture(body_box_list, kps_list, hand_box_list, hand_lmk_list,
                gesture_list);
}

void MergeHandBody::FilterHandGesture(xstream::OutputDataPtr xstream_out,
                                      bool merge_hand_id) {
  xstream::BaseDataVector *kps_list = nullptr;
  xstream::BaseDataVector *hand_lmk_list = nullptr;
  xstream::BaseDataVector *body_box_list = nullptr;
  xstream::BaseDataVector *hand_box_list = nullptr;
  xstream::BaseDataVector *gesture_list = nullptr;
  xstream::BaseDataVector *face_boxe_list = nullptr;
  auto name_prefix = [](const std::string name) -> std::string {
    auto pos = name.find('_');
    if (pos == std::string::npos)
      return "";

    return name.substr(0, pos);
  };
  auto name_postfix = [](const std::string name) -> std::string {
    auto pos = name.rfind('_');
    if (pos == std::string::npos)
      return "";

    return name.substr(pos + 1);
  };
  for (const auto &output : xstream_out->datas_) {
    LOGD << output->name_ << ", type is " << output->type_;
    auto prefix = name_prefix(output->name_);
    auto postfix = name_postfix(output->name_);
    if (output->name_ == "kps" || output->name_ == "lowpassfilter_body_kps") {
      kps_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "hand_lmk" ||
        output->name_ == "lowpassfilter_hand_lmk") {
      hand_lmk_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
    if (output->name_ == "body_box" || prefix == "body") {
      body_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
    if (prefix == "hand" && postfix == "box") {
      hand_box_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }

    if (output->name_ == "gesture_vote") {
      gesture_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
    if (output->name_ == "face_bbox_list") {
      face_boxe_list = dynamic_cast<xstream::BaseDataVector *>(output.get());
    }
  }

  UpdateHandIDGesture(body_box_list, kps_list, hand_box_list, hand_lmk_list,
                      face_boxe_list, gesture_list, merge_hand_id);
}

float Float_Min(float x, float y) {
  if (x > y && fabs(x - y) > 1e-6) {
    return y;
  } else {
    return x;
  }
}

float CrossProduct(const hobot::vision::Point a, const hobot::vision::Point b,
                   hobot::vision::Point c) {
  float ab = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
  float ac = sqrt((a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y));
  float bc = sqrt((b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y));
  float p = (ab + ac + bc) / 2;
  float s = sqrt(p * (p - ab) * (p - ac) * (p - bc));
  return s / ab;
}

bool MergeHandBody::UpdateHandID(
    xstream::BaseDataVector *face_boxes_list,
    std::shared_ptr<xstream::XStreamData<hobot::vision::Landmarks>> hand_lmk,
    int32_t &new_track_id) {
  int32_t old_track_id = new_track_id;
  const auto &point = hand_lmk->value.values[0];
  for (size_t i = 0; i < face_boxes_list->datas_.size(); ++i) {
    auto face_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            face_boxes_list->datas_[i]);
    if (!face_box) continue;

    if (face_box->value.score <= 0.0) {
      continue;
    }

    if (point.x >= face_box->value.x1 && point.x <= face_box->value.x2 &&
        point.y >= face_box->value.y1 && point.y <= face_box->value.y2) {
      new_track_id = face_box->value.id;
      LOGD << "update hand id hand key point in face box, old "
              "trackid:"
           << old_track_id << ", new trackid:" << new_track_id;
      return true;
    }
  }
  LOGD << "update hand id hand key point not in face box, old "
          "trackid:"
       << old_track_id << ", new trackid:" << new_track_id;
  return false;
}

bool MergeHandBody::IsHandOnOneLine(
    std::shared_ptr<xstream::XStreamData<hobot::vision::Landmarks>> hand_lmk,
    std::shared_ptr<xstream::XStreamData<hobot::vision::Landmarks>> lmk,
    int32_t body_trackid, const bool left_hand) {
  auto &point1 = lmk->value.values[7];
  auto &point2 = lmk->value.values[9];
  auto &point3 = lmk->value.values[5];
  if (!left_hand) {
    point1 = lmk->value.values[8];
    point2 = lmk->value.values[10];
    point3 = lmk->value.values[6];
  }

  const auto &hand_point = hand_lmk->value.values[0];
  const auto &hand_point1 = hand_lmk->value.values[9];
  constexpr float eps = 50.0;  // 1e-8;
  if (point1.score > 0) {
    float distance = CrossProduct(point1, point2, hand_point);
    if (distance > eps && fabs(distance - eps) > 1e-1) {
      return false;
    }
    LOGW << "update hand id body trackid:" << body_trackid
         << ", arm key point on line, distance:"
         << distance << ", hand key point x:"
         << hand_point.x << ", y:" << hand_point.y
         << ", arm point x:" << point1.x
         << ", y:" << point1.y << ",   x:" << point2.x
         << ", y:" << point2.y << "  left hand flag:" << left_hand;
    distance = CrossProduct(point3, hand_point1, hand_point);
    if (distance > eps && fabs(distance - eps) > 1e-1) {
      return false;
    }

    LOGW << "update hand id body trackid:" << body_trackid
         << ", arm key point and hand "
            "point10, key point on line, "
            "distance:"
         << distance << ", arm point x:" << point3.x << ", y:" << point3.y
         << ",  hand 10 x:" << hand_point1.x << ", y:" << hand_point1.y
         << ", hand 1 x:" << hand_point.x << ", y:" << hand_point.y
         << "  left hand flag:" << left_hand;
    distance = sqrt((point1.x - point2.x) * (point1.x - point2.x) +
                    (point1.y - point2.y) * (point1.y - point3.y));
    float tmp = sqrt((point1.x - hand_point.x) * (point1.x - hand_point.x) +
                     (point1.y - hand_point.y) * (point1.y - hand_point.y));
    if (tmp > distance && (tmp - distance) > 1e-3) {
      return true;  // The hand is in a straight state, and key points are in
                    // order
    }
  }  // end of point1 score > 0
  return false;
}

bool MergeHandBody::UpdateDistance(
    std::shared_ptr<xstream::XStreamData<hobot::vision::Landmarks>> hand_lmk,
    std::shared_ptr<xstream::XStreamData<hobot::vision::Landmarks>> kps,
    std::shared_ptr<xstream::XStreamData<hobot::vision::BBox>> body_box,
    float &min_distance, int32_t &new_track_id) {
  if (hand_lmk->value.values.size() <= 0) {
    LOGE << "MergeHandBody recv hand lmk point size is 0";
    return false;
  }

  const auto &point9 = kps->value.values[9];
  const auto &point10 = kps->value.values[10];
  if (point9.score < 0 && point10.score < 0) {
    LOGW << "update hand id find key point score less than 0, return";
    return false;
  }

  bool update = false;
  const auto &point = hand_lmk->value.values[0];
  float distance_9 = sqrt((point9.x - point.x) * (point9.x - point.x) +
                          (point9.y - point.y) * (point9.y - point.y));
  float distance_10 = sqrt((point10.x - point.x) * (point10.x - point.x) +
                           (point10.y - point.y) * (point10.y - point.y));
  float tmp_distance = Float_Min(distance_9, distance_10);
  if (min_distance > tmp_distance && fabs(tmp_distance - min_distance) > 1e-6) {
    min_distance = tmp_distance;
    new_track_id = body_box->value.id;
    if (tmp_distance <= 100.0) update = true;
  }

  // if(point9.score > 0) {
  if (distance_9 < distance_10 && fabs(distance_9 - distance_10) < 1e-3) {
    if (IsHandOnOneLine(hand_lmk, kps, body_box->value.id)) {  // on one line
      tmp_distance =
          CrossProduct(kps->value.values[5], kps->value.values[7], point9);
      if (tmp_distance < 50.0 &&
          fabs(tmp_distance - 50.0) > 1e-1) {  // one line
        if (CrossProduct(kps->value.values[7],
           point9, kps->value.values[11]) < 100.0)
          return true;
      }

      if (kps->value.values[5].x <= kps->value.values[7].x &&
          fabs(kps->value.values[5].x - kps->value.values[7].x) < 1e-1) {
        // if (point9.x < point.x && fabs(point9.x - point.x) < 1e-1) {
        if (point9.x < hand_lmk->value.values[9].x && update) {
          return true;
        }
        return false;
      } else {
        // if (point9.x > point.x && fabs(point9.x - point.x) < 1e-1) {
        if (point9.x > hand_lmk->value.values[9].x && update) {
          return true;
        }
        return false;
      }
    } else {  // not on one line
      // todo
    }
  } else if (distance_10 < distance_9 &&
             fabs(distance_9 - distance_10) < 1e-3) {
    if (IsHandOnOneLine(hand_lmk, kps, body_box->value.id, false)) {
      // on one line
      tmp_distance =
          CrossProduct(kps->value.values[6], kps->value.values[8], point10);
      if (tmp_distance < 50.0 &&
          fabs(tmp_distance - 50.0) > 1e-1) {  // one line
        if (CrossProduct(kps->value.values[8],
           point10, kps->value.values[12]) < 100.0)
          return true;
      }

      if (kps->value.values[6].x <= kps->value.values[8].x &&
          fabs(kps->value.values[6].x - kps->value.values[8].x) < 1e-1) {
        // if (point10.x < point.x && fabs(point10.x - point.x) < 1e-1) {
        if (point10.x < hand_lmk->value.values[9].x && update) {
          return true;
        }
        return false;
      } else {
        // if (point10.x > point.x && fabs(point10.x - point.x) < 1e-1) {
        if (point10.x > hand_lmk->value.values[9].x && update) {
          return true;
        }
        return false;
      }
    } else {  // not on one line
      // todo
    }
  } else {
  }

  return false;
}

bool MergeHandBody::UpdateHandID(
    xstream::BaseDataVector *body_list, xstream::BaseDataVector *kps_list,
    std::shared_ptr<xstream::XStreamData<hobot::vision::Landmarks>> hand_lmk,
    int32_t &new_track_id) {
  int32_t old_track_id = new_track_id;
  float min_distance = 10000.0;
  for (size_t j = 0; j < kps_list->datas_.size(); ++j) {
    auto lmk = std::static_pointer_cast<
        xstream::XStreamData<hobot::vision::Landmarks>>(kps_list->datas_[j]);
    if (!lmk) continue;

    if (lmk->value.values.size() < 17) {
      LOGE << "UpdateHandTrackID body kps size less than 17, error!!!";
      continue;
    }
    auto body_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            body_list->datas_[j]);
    if (!body_box) continue;
    if (body_box->value.id == -1) {
      continue;
    }

    bool same_hand = false;
    same_hand =
        UpdateDistance(hand_lmk, lmk, body_box, min_distance, new_track_id);
    if (same_hand) {
      new_track_id = body_box->value.id;
      return true;
    }
  }  // end of for

  if (min_distance > 500.0) {
    LOGI << "update hand id find hand track:" << old_track_id
         << " the min key point distance more than 500, distance:"
         << min_distance << ",return false ";
    return false;
  }
  return true;
}

void MergeHandBody::UpdateHandTrackID(xstream::BaseDataVector *body_list,
                                      xstream::BaseDataVector *kps_list,
                                      xstream::BaseDataVector *hand_box_list,
                                      xstream::BaseDataVector *hand_lmk_list,
                                      xstream::BaseDataVector *face_boxe_list) {
  if (body_list == NULL || kps_list == NULL || hand_box_list == NULL ||
      hand_lmk_list == NULL) {
    LOGD << "UpdateHandTrackID recv null pointer";
    return;
  }

  if (hand_lmk_list->datas_.size() != hand_box_list->datas_.size()) {
    LOGE << "UpdateHandTrackID recv hand lmk list size != hand box list size";
    return;
  }
  if (body_list->datas_.size() != kps_list->datas_.size()) {
    LOGE << "UpdateHandTrackID recv body box list size != kps list size";
    return;
  }

  for (size_t i = 0; i < hand_lmk_list->datas_.size(); ++i) {
    auto hand_lmk = std::static_pointer_cast<
        xstream::XStreamData<hobot::vision::Landmarks>>(
        hand_lmk_list->datas_[i]);
    if (!hand_lmk) continue;

    // 查找对应的track_id
    auto hand_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            hand_box_list->datas_[i]);
    if (!hand_box || hand_box->value.id == -1) {
      continue;
    }

    int32_t new_track_id = hand_box->value.id;
    if (UpdateHandID(body_list, kps_list, hand_lmk, new_track_id)) {
      LOGD << "update hand id to body id, old track:" << hand_box->value.id
           << " new track:" << new_track_id;
      hand_box->value.id = new_track_id;
    } else {
      if (face_boxe_list) {
        if (UpdateHandID(face_boxe_list, hand_lmk, new_track_id)) {
          LOGD << "update hand id to face id, old track:" << hand_box->value.id
               << " new track:" << new_track_id;
          hand_box->value.id = new_track_id;
        }
      }
    }  // end of else
  }    // end of for
}

void MergeHandBody::FilterGesture(xstream::BaseDataVector *body_list,
                                  xstream::BaseDataVector *kps_list,
                                  xstream::BaseDataVector *hand_box_list,
                                  xstream::BaseDataVector *hand_lmk_list,
                                  xstream::BaseDataVector *gesture_list) {
  if (body_list == NULL || kps_list == NULL || hand_box_list == NULL ||
      hand_lmk_list == NULL || gesture_list == NULL) {
    LOGD << "FilterGesture recv null pointer";
    return;
  }
  if (hand_lmk_list->datas_.size() != hand_box_list->datas_.size()) {
    LOGE << "FilterGesture recv hand lmk list size != hand box list "
            "size";
    return;
  }

  if (gesture_list->datas_.size() != hand_box_list->datas_.size()) {
    LOGE << "FilterGesture recv gesture list size != hand box list "
            "size";
    return;
  }

  if (body_list->datas_.size() != kps_list->datas_.size()) {
    LOGE << "FilterGesture recv body box list size != kps list size";
    return;
  }

  constexpr float eps = 100.0;
  for (size_t i = 0; i < hand_lmk_list->datas_.size(); ++i) {
    auto hand_lmk = std::static_pointer_cast<
        xstream::XStreamData<hobot::vision::Landmarks>>(
        hand_lmk_list->datas_[i]);
    if (!hand_lmk)
      continue;

    // 查找对应的track_id
    auto hand_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            hand_box_list->datas_[i]);
    if (!hand_box)
      continue;
    if (hand_box->value.id == -1) {
      continue;
    }
    const auto &point = hand_lmk->value.values[0];
    int32_t track_id = hand_box->value.id;
    for (size_t j = 0; j < body_list->datas_.size(); ++j) {
      auto body_box =
          std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
              body_list->datas_[j]);
      if (!body_box) {
        continue;
      }
      if (body_box->value.id != track_id) {
        continue;
      }

      auto lmk = std::static_pointer_cast<
          xstream::XStreamData<hobot::vision::Landmarks>>(kps_list->datas_[j]);
      if (!lmk)
       break;
      const auto &point11 = lmk->value.values[11];
      const auto &point12 = lmk->value.values[12];
      if (((point11.y < point.y + eps) &&
           fabs(point11.y - point.y - eps) > 1e-6) ||
          ((point12.y < point.y + eps) &&
           fabs(point12.y - point.y - eps) > 1e-6)) {
        auto gesture_vote = std::static_pointer_cast<
            xstream::XStreamData<hobot::vision::Attribute<int>>>(
            gesture_list->datas_[i]);
        if (gesture_vote->state_ == xstream::DataState::VALID) {
          gesture_vote->value.value = 0;
        }
      }
      break;
    }
  }
}

void MergeHandBody::UpdateHandIDGesture(xstream::BaseDataVector *body_list,
                                      xstream::BaseDataVector *kps_list,
                                      xstream::BaseDataVector *hand_box_list,
                                      xstream::BaseDataVector *hand_lmk_list,
                                      xstream::BaseDataVector *face_boxe_list,
                                      xstream::BaseDataVector *gesture_list,
                                      bool merge_id) {
  if (body_list == NULL || kps_list == NULL || hand_box_list == NULL ||
      hand_lmk_list == NULL) {
    LOGD << "UpdateHandTrackID recv null pointer";
    return;
  }

  if (hand_lmk_list->datas_.size() != hand_box_list->datas_.size()) {
    LOGE << "UpdateHandTrackID recv hand lmk list size != hand box list size";
    return;
  }
  if (body_list->datas_.size() != kps_list->datas_.size()) {
    LOGE << "UpdateHandTrackID recv body box list size != kps list size";
    return;
  }
  if (gesture_list->datas_.size() != hand_box_list->datas_.size()) {
    LOGE << "FilterGesture recv gesture list size != hand box list "
            "size";
    return;
  }

  for (size_t i = 0; i < hand_box_list->datas_.size(); ++i) {
    auto hand_lmk = std::static_pointer_cast<
        xstream::XStreamData<hobot::vision::Landmarks>>(
        hand_lmk_list->datas_[i]);
    if (!hand_lmk)
      continue;

    auto hand_box =
        std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
            hand_box_list->datas_[i]);
    if (!hand_box || hand_box->value.id == -1) {
      continue;
    }

    float dis_max = std::numeric_limits<float>::max();
    int index = -1;
    bool left_hand;
    for (size_t j = 0; j < kps_list->datas_.size(); ++j) {
      auto body_kps = std::static_pointer_cast<
          xstream::XStreamData<hobot::vision::Landmarks>>(kps_list->datas_[j]);
      if (!body_kps)
        continue;
      if (body_kps->value.values.size() < 19) {
        LOGE << "UpdateHandTrackID body kps size less than 19, error!!!";
        continue;
      }

      auto hand_box_centre_x = (hand_box->value.x1 + hand_box->value.x2) / 2;
      auto hand_box_centre_y = (hand_box->value.y1 + hand_box->value.y2) / 2;
      auto body_kps_hand_x = body_kps->value.values[17].x;
      auto body_kps_hand_y = body_kps->value.values[17].y;
      if (body_kps_hand_x >= hand_box->value.x1 &&
          body_kps_hand_x <= hand_box->value.x2 &&
          body_kps_hand_y >= hand_box->value.y1 &&
          body_kps_hand_y <= hand_box->value.y2) {
        auto dis = (body_kps_hand_x - hand_box_centre_x) *
                       (body_kps_hand_x - hand_box_centre_x) +
                   (body_kps_hand_y - hand_box_centre_y) *
                       (body_kps_hand_y - hand_box_centre_y);
        if (dis < dis_max) {
          index = j;
          dis_max = dis;
          left_hand = true;
        }
      }
      body_kps_hand_x = body_kps->value.values[18].x;
      body_kps_hand_y = body_kps->value.values[18].y;
      if (body_kps_hand_x >= hand_box->value.x1 &&
          body_kps_hand_x <= hand_box->value.x2 &&
          body_kps_hand_y >= hand_box->value.y1 &&
          body_kps_hand_y <= hand_box->value.y2) {
        auto dis = (body_kps_hand_x - hand_box_centre_x) *
                       (body_kps_hand_x - hand_box_centre_x) +
                   (body_kps_hand_y - hand_box_centre_y) *
                       (body_kps_hand_y - hand_box_centre_y);
        if (dis < dis_max) {
          index = j;
          dis_max = dis;
          left_hand = false;
        }
      }
    }

    if (index == -1) {
      // hand框中没有找到对应的kps手点
      LOGI << "Did not find body box for hand box!";
    } else {
      // 更新hand box id
      if (merge_id) {
        auto body_box =
            std::static_pointer_cast<xstream::XStreamData<hobot::vision::BBox>>(
                body_list->datas_[index]);
        if (!body_box)
          continue;
        if (body_box->value.id == -1) {
          continue;
        }
        hand_box->value.id = body_box->value.id;
      }

      // 过滤腰线以下手势
      auto body_kps = std::static_pointer_cast<
          xstream::XStreamData<hobot::vision::Landmarks>>(
          kps_list->datas_[index]);
      if (!body_kps)
        continue;
      if (body_kps->value.values.size() < 19) {
        LOGE << "UpdateHandTrackID body kps size less than 19, error!!!";
        continue;
      }
      if (left_hand) {
        if (body_kps->value.values[17].y > body_kps->value.values[12].y) {
          auto gesture_vote = std::static_pointer_cast<
              xstream::XStreamData<hobot::vision::Attribute<int>>>(
              gesture_list->datas_[i]);
          if (gesture_vote->state_ == xstream::DataState::VALID) {
            gesture_vote->value.value = 0;
          }
        }
      } else {
        if (body_kps->value.values[18].y > body_kps->value.values[11].y) {
          auto gesture_vote = std::static_pointer_cast<
              xstream::XStreamData<hobot::vision::Attribute<int>>>(
              gesture_list->datas_[i]);
          if (gesture_vote->state_ == xstream::DataState::VALID) {
            gesture_vote->value.value = 0;
          }
        }
      }
    }
  }
}

}  // namespace smartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
