/*!
 * Copyright (c) 2016-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     aioplugin_data.h
 * \Author   ronghui.zhang
 * \Version  1.0.0.0
 * \Date     2021.1.7
 */
#ifndef XPROTO_MSGTYPE_AIOPLUGIN_DATA_H_
#define XPROTO_MSGTYPE_AIOPLUGIN_DATA_H_

#include <string>
#include "horizon/vision_type/vision_error.h"
#include "horizon/vision_type/vision_msg.h"
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/utils/profile.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace basic_msgtype {

#define TYPE_AUDIO_MESSAGE "XPLUGIN_AUDIO_MESSAGE"

struct AioMessage : public XProtoMessage {
 public:
  AioMessage() {
    type_ = TYPE_AUDIO_MESSAGE;
  }
  explicit AioMessage(char* buffer, int size, int num) {
    type_ = TYPE_AUDIO_MESSAGE;
    buffer_ = buffer;
    size_ = size;
    num_ = num;
  }
  virtual ~AioMessage() = default;

  // audio frames number
  uint32_t num_ = 0;
  // sequence id, would increment automatically
  uint64_t sequence_id_ = 0;
  // time stamp
  uint64_t time_stamp_ = 0;
  // is valid uri
  bool is_valid_uri_ = true;

  // serialize proto
  std::string Serialize() override { return "Default aio message"; };
  // audio data
 public:
  // free source image
  void FreeAudio() {
    if (buffer_) {
      free(buffer_);
    }
  }
  char* buffer_;
  int size_;
};

}  // namespace basic_msgtype
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif
