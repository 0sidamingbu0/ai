/*
 * Copyright (c) 2020-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     smart_manager.h
 * \Author   xudong.du
 * \Version  1.0.0.0
 * \Date     2020/9/27
 * \Brief    implement of api header
 */
#ifndef INCLUDE_UVCPLUGIN_SMARTMANAGER_H_
#define INCLUDE_UVCPLUGIN_SMARTMANAGER_H_
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "json/json.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto_msgtype/smartplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace Uvcplugin {
using horizon::vision::xproto::XProtoMessagePtr;

class SmartManager {
 public:
  SmartManager() = default;
  virtual ~SmartManager() = default;
  virtual int Init() { return 0; }
  virtual int DeInit() { return 0; }
  virtual int Start() { return 0; }
  virtual int Stop() { return 0; }

  virtual int FeedSmart(XProtoMessagePtr msg, int ori_image_width,
                        int ori_image_height, int dst_image_width,
                        int dst_image_height) {
    return 0;
  }
  virtual int FeedDropSmart(uint64_t frame_id) {
    return 0;
  }
};

}  // namespace Uvcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // INCLUDE_UVCPLUGIN_SMARTMANAGER_H_
