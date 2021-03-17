/*
 * @Description: implement of vioplugin
 * @Author: fei.cheng@horizon.ai
 * @Date: 2019-08-26 16:17:25
 * @Author: songshan.gong@horizon.ai
 * @Date: 2019-09-26 16:17:25
 * @LastEditors: hao.tian@horizon.ai
 * @LastEditTime: 2019-10-15 11:14:07
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#ifndef INCLUDE_VIOPLUGIN_VIOPLUGIN_H_
#define INCLUDE_VIOPLUGIN_VIOPLUGIN_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "json/json.h"
#include "vioplugin/vioproduce.h"
#include "xproto/plugin/xpluginasync.h"

#include "horizon/vision_type/vision_type.hpp"
#include "hobot_vision/blocking_queue.hpp"

#define VIO_CAMERA "vio_camera"
#define VIO_FEEDBACK "vio_feedback"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

using box_t = hobot::vision::BBox_<uint32_t>;
using Listener = std::function<int(const std::shared_ptr<VioMessage> &input)>;

class VioPlugin : public xproto::XPluginAsync {
 public:
  VioPlugin() = delete;
  explicit VioPlugin(const std::string &path);
  ~VioPlugin() override;
  int Init() override;
  int DeInit() override;
  int Start() override;
  int Stop() override;
  int CamStart();
  int FbStart();
  int CamStop();
  int FbStop();
  std::string desc() const { return "VioPlugin"; }
  bool IsInited() { return is_inited_; }
  XProtoMessagePtr GetImage();
  void ClearAllQueue();
  int SetMode(bool is_sync_mode) {
    is_sync_mode_ = is_sync_mode;
    return 0;
  }

 private:
  std::shared_ptr<VioConfig> GetConfigFromFile(const std::string &path);
  int OnGetHbipcResult(const XProtoMessagePtr msg);
  void GetSubConfigs();
  Listener SetListenerFunc();

 private:
  std::shared_ptr<VioConfig> config_;
  std::vector<std::shared_ptr<VioConfig>> configs_;
  std::vector<std::shared_ptr<VioProduce>> vio_produce_handles_;
  std::vector<box_t> Shields_;
  bool is_inited_ = false;
  bool is_sync_mode_ = false;
  bool is_running_ = false;
  hobot::vision::BlockingQueue<XProtoMessagePtr> img_msg_queue_;
  hobot::vision::BlockingQueue<XProtoMessagePtr> drop_msg_queue_;
  std::unordered_map<std::string, XProtoMessageFunc> message_cb_;
  int vio_config_num_ = 0;
#ifdef USE_MC
  int OnGetAPImage(XProtoMessagePtr msg);
#endif
};

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif
