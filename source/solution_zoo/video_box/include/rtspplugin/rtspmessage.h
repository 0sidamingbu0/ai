/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail: @horizon.ai
 */
#ifndef INCLUDE_RTSPMESSAGE_RTSPMESSAGE_H_
#define INCLUDE_RTSPMESSAGE_RTSPMESSAGE_H_

#include <memory>
#include <vector>

#include "hobot_vision/blocking_queue.hpp"

#include "mediapipemanager/meidapipeline.h"
#include "xproto_msgtype/vioplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace rtspplugin {
using hobot::vision::PymImageFrame;
using horizon::vision::xproto::basic_msgtype::VioMessage;

struct ImageVioMessage : VioMessage {
public:
  ImageVioMessage() = delete;
  explicit ImageVioMessage(
      std::vector<std::shared_ptr<PymImageFrame>> &image_frame,
      uint32_t img_num, bool is_valid = true, int channel = -1,
      std::shared_ptr<MediaPipeLine> pipeline = nullptr,
      void *data = nullptr, const int index = 0);
  ~ImageVioMessage();

  // serialize proto
  std::string Serialize() { return "No need serialize"; };

  void FreeImage();
  void FreeImage(int tmp); // 用于释放x3临时回灌功能的接口

  std::shared_ptr<MediaPipeLine> pipeline_;
  void *slot_data_;
  int index_;
};

struct DropVioMessage : VioMessage {
public:
  DropVioMessage() = delete;
  explicit DropVioMessage(uint64_t timestamp, uint64_t seq_id);
  ~DropVioMessage(){};

  // serialize proto
  std::string Serialize() override;
};

}  // namespace rtspplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif
