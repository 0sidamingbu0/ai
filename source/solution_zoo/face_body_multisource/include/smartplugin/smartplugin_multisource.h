/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Songshan Gong
 * @Mail: songshan.gong@horizon.ai
 * @Date: 2019-08-04 02:41:22
 * @Version: v0.0.1
 * @Brief: smartplugin declaration
 * @Last Modified by: Songshan Gong
 * @Last Modified time: 2019-09-30 00:45:01
 */

#ifndef INCLUDE_MULTISOURCE_SMARTPLUGIN_SMARTPLUGIN_H_
#define INCLUDE_MULTISOURCE_SMARTPLUGIN_SMARTPLUGIN_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "multisourcesmartplugin/smartplugin.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/message/pluginflow/msg_registry.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {
#define FACE_BODY_SMART_MESSAGE "FACE_BODY_XPLUGIN_SMART_MESSAGE"
using horizon::vision::xproto::multisourcesmartplugin::CustomSmartMessage;
using horizon::vision::xproto::multisourcesmartplugin::SmartPlugin;
using xstream::OutputDataPtr;

XPLUGIN_REGISTER_MSG_TYPE(FACE_BODY_XPLUGIN_SMART_MESSAGE)

struct target_key {
  std::string category;
  int id;
  target_key(std::string category, int id) {
    this->category = category;
    this->id = id;
  }
};

struct cmp_key {
  bool operator()(const target_key &a, const target_key &b) {
    if (a.category == b.category) {
      return a.id < b.id;
    }
    return a.category < b.category;
  }
};

struct FaceBodySmartMessage : CustomSmartMessage {
  explicit FaceBodySmartMessage(xstream::OutputDataPtr out)
      : CustomSmartMessage(out) {
    std::cout << "explicit FaceBodySmartMessage" << std::endl;
    type_ = FACE_BODY_SMART_MESSAGE;
  }

  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) override;
};

class FaceBodySmartPlugin : public SmartPlugin {
 public:
  explicit FaceBodySmartPlugin(const std::string& config_file);

 private:
  std::string GetWorkflowInputImageName() { return "image"; }

  std::shared_ptr<CustomSmartMessage> CreateSmartMessage(
      xstream::OutputDataPtr xstream_out) {
    return std::make_shared<FaceBodySmartMessage>(xstream_out);
  }
};

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_MULTISOURCE_SMARTPLUGIN_SMARTPLUGIN_H_
