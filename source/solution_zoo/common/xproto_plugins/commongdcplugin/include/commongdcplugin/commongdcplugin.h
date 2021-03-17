/*
 * @Description: implement of reorderplugin.h
 * @Author: xudong.du@horizon.ai
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#ifndef COMMON_GDCPLUGIN_INCLUDE_COMMON_GDCPLUGIN_H_
#define COMMON_GDCPLUGIN_INCLUDE_COMMON_GDCPLUGIN_H_

#include <chrono>
#include <memory>
#include <string>
#include <queue>
#include <vector>

/* dependency header */
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto/threads/threadpool.h"

#include "xproto_msgtype/commongdcplugin_data.h"
#include "xproto_msgtype/hbipcplugin_data.h"
#include "xproto_msgtype/smartplugin_data.h"
#include "xproto_msgtype/vioplugin_data.h"
#include "commongdcplugin/gdcmanager.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

using horizon::vision::xproto::basic_msgtype::CustomGdcMessage;
using horizon::vision::xproto::basic_msgtype::SmartMessage;
using horizon::vision::xproto::basic_msgtype::VioMessage;

class CommonGdcPlugin : public XPluginAsync {
 public:
  CommonGdcPlugin() = default;
  explicit CommonGdcPlugin(std::string cfg_file);
  ~CommonGdcPlugin();

 public:
  /* xproto框架接口的封装函数 */
  // 初始化plugin
  int Init() override;
  // 反初始化plugin
  int DeInit();
  // 开启plugin服务
  int Start() override;
  // 关闭plugin服务
  int Stop() override;
  // 返回plugin的名称
  std::string desc() const { return "CommonGdcPlugin"; }

 private:
  // 接收vio消息
  int OnGetVioMessage(const XProtoMessagePtr msg);
  void GetGdcResult(const XProtoMessagePtr msg);
  int ParseConfig();

 private:
  std::string config_file_;
  Json::Value config_;
  hobot::CThreadPool gdc_work_thread_;
  std::atomic<bool> is_stop_;
  std::shared_ptr<GdcConfig> gdc_cfg_;
};

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // COMMON_GDCPLUGIN_INCLUDE_COMMON_GDCPLUGIN_H_
