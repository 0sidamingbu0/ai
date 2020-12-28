/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file ExampleSmartplugin.h
 * @brief
 * @author ronghui.zhang
 * @email ronghui.zhang@horizon.ai
 *
 *
 * */
#ifndef INCLUDE_EXAMPLESMARTPLUGIN_EXAMPLESMARTPLUGIN_H_
#define INCLUDE_EXAMPLESMARTPLUGIN_EXAMPLESMARTPLUGIN_H_

#include <string>
#include <memory>
#include "smartplugin/smartplugin.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/message/pluginflow/msg_registry.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace ExampleSmartPlugin {

#define EXAMPLE_SMART_MESSAGE "EXAMPLE_XPLUGIN_SMART_MESSAGE"
using horizon::vision::xproto::smartplugin::SmartPlugin;
using horizon::vision::xproto::smartplugin::CustomSmartMessage;
using xstream::OutputDataPtr;

XPLUGIN_REGISTER_MSG_TYPE(EXAMPLE_XPLUGIN_SMART_MESSAGE)

struct ExampleCustomSmartMessage : CustomSmartMessage {
  explicit ExampleCustomSmartMessage(
    xstream::OutputDataPtr out) : CustomSmartMessage(out) {
    std::cout << "explicit ExampleCustomSmartMessage" << std::endl;
    type_ = EXAMPLE_SMART_MESSAGE;
  }

  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) override;
};

class ExampleSmartPlugin : public SmartPlugin {
 public:
  explicit ExampleSmartPlugin(const std::string& config_file);

 private:
  std::string GetWorkflowInputImageName() {
    return "image";
  }

  std::shared_ptr<CustomSmartMessage>
  CreateSmartMessage(xstream::OutputDataPtr xstream_out) {
    return std::make_shared<ExampleCustomSmartMessage>(xstream_out);
  }
};
}  // namespace ExampleSmartPlugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif
