/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCESMARTPLUGIN_SMARTPLUGIN_H_
#define INCLUDE_MULTISOURCESMARTPLUGIN_SMARTPLUGIN_H_

#include <memory>
#include <string>
#include <map>
#include <vector>

#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"

#include "hobotxsdk/xstream_sdk.h"
#include "multisourcesmartplugin/runtime_monitor.h"
#include "multisourcesmartplugin/smart_config.h"
#include "vioplugin/viomessage.h"

#include "xproto_msgtype/smartplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;

using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;
using horizon::vision::xproto::basic_msgtype::VioMessage;
using horizon::vision::xproto::basic_msgtype::SmartMessage;

struct CustomSmartMessage : SmartMessage {
  explicit CustomSmartMessage(xstream::OutputDataPtr out)
      : smart_result_(out) {
    type_ = TYPE_SMART_MESSAGE;
  }
  std::string Serialize() override;
  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) override;
  void *ConvertData() override;
 protected:
  xstream::OutputDataPtr smart_result_;
};

class SmartPlugin : public XPluginAsync {
 public:
  SmartPlugin() = default;
  explicit SmartPlugin(const std::string &config_file);

  void SetConfig(const std::string &config_file) { config_file_ = config_file; }

  ~SmartPlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;

 private:
  // 获取单路图像，workflow配置的图像输入节点名字，根据workflow_idx区分
  // workflow_idx从0开始计数0,1,2...
  // SmartPlugin派生类可以根据需要修改输入节点的名字
  // 但是必须保证该接口返回的图像输入节点名字和xstream json配置中一致
  virtual std::string GetWorkflowInputImageName(uint32_t workflow_idx) {
    std::string input_name = "image";  // default
    if (workflow_idx >= 0) {
      input_name = "image";
    }
    return input_name;
  }
  // 创建xproto框架下感知结果的消息对象
  // 感知结果消息对象必须是CustomSmartMessage或者集成自CustomSmartMessage
  // 输入参数xstream_out为xstream workflow执行完成，xstream回调返回的数据对象
  virtual std::shared_ptr<CustomSmartMessage> CreateSmartMessage(
      xstream::OutputDataPtr xstream_out) {
    return std::make_shared<CustomSmartMessage>(xstream_out);
  }

  int Feed(XProtoMessagePtr msg);
  int FeedIpm(XProtoMessagePtr msg);
  int FeedMulti(XProtoMessagePtr msg);
  void OnCallback(xstream::OutputDataPtr out);
  void ParseConfig();
  int OverWriteSourceNum(const std::string &cfg_file, int source_num = 1);

  std::vector<std::shared_ptr<XStreamSDK>> sdk_;
  std::shared_ptr<RuntimeMonitor> monitor_;
  std::string config_file_;
  // source id => [target workflow, ...]
  std::map<int, std::vector<int>> source_target_;
  // xstream instance => input source list
  std::vector<std::vector<int>> source_map_;

  std::shared_ptr<JsonConfigWrapper> config_;
};

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_MULTISOURCESMARTPLUGIN_SMARTPLUGIN_H_
