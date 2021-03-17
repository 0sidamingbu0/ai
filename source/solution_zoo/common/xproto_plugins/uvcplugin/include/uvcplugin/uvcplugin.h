/*!
 * Copyright (c) 2016-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     uvcplugin.h
 * \Author   ronghui.zhang
 * \Version  1.0.0.0
 * \Date     2020/5/12
 * \Brief    implement of api header
 */
#ifndef INCLUDE_UVCPLUGIN_UVCPLUGIN_H_
#define INCLUDE_UVCPLUGIN_UVCPLUGIN_H_
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "./smart_manager.h"
#include "./hid_manager.h"
#include "./rndis_manager.h"
#include "./usb_common.h"
#include "guvc/uvc.h"
#include "guvc/uvc_gadget.h"
// undef problematic defines sometimes defined by system headers (uvc_gadget.h)
#undef max
#include "./uvc_server.h"
#include "./uac_server.h"
#include "./uvcplugin_config.h"
#include "./venc_client.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "uevent_helper.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace Uvcplugin {
using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessagePtr;

class UvcPlugin : public xproto::XPluginAsync {
 public:
  UvcPlugin() = delete;
  explicit UvcPlugin(std::string config_path);
  ~UvcPlugin() override;
  int Init() override;
  int DeInit() override;
  int Start() override;
  int Stop() override;
  std::string desc() const { return "uvcPlugin"; }

 private:
  int FeedVideoMsg(XProtoMessagePtr msg);
  int FeedVideoDropMsg(XProtoMessagePtr msg);
  int FeedVideo(XProtoMessagePtr msg);
  int FeedSmart(XProtoMessagePtr msg);
  int FeedVideoDrop(XProtoMessagePtr msg);
  int FeedMc(XProtoMessagePtr msg);
  int FeedMicphoneAudioMsg(XProtoMessagePtr msg);
  int FeedMicphoneAudio(XProtoMessagePtr msg);
  void ParseConfig();
  int Reset();
  int ParseConfig(std::string config_file);

  void Dwc3UeventMonitor();
  void MonitorEvent();
  int ReInit();
  int ReStart();
  int StopUvcHid();

 private:
  int smart_transfer_mode_ = 0;  // 0: hid, 1: rndis, default 0
  std::string config_file_;
  bool run_flag_;
  std::shared_ptr<std::thread> worker_;
  std::mutex map_mutex_;
  const uint8_t cache_size_ = 25;  // max input cache size
  std::shared_ptr<UvcServer> uvc_server_;
  std::shared_ptr<VencClient> venc_client_;
  struct uvc_context *uvc_ctx;
  std::shared_ptr<UvcConfig> config_;

  std::shared_ptr<std::thread> thread_process_ap_ = nullptr;
  std::shared_ptr<SmartManager> smart_manager_;

  std::shared_ptr<std::thread> monitor_thread_;
  std::shared_ptr<std::thread> dwc3_thread_;

  std::shared_ptr<UacServer> uac_server_;

  int origin_image_width_ = 1920;  // update by FeedVideo
  int origin_image_height_ = 1080;
  int dst_image_width_ = 1920;  // update by FeedVideo
  int dst_image_height_ = 1080;
  VIDEO_STREAM_S h264_sps_frame_;

  std::mutex video_send_mutex_;
  int video_sended_without_recv_count_;
  hobot::CThreadPool encode_thread_;
  hobot::CThreadPool send_audio_thread_;
  bool print_timestamp_ = false;
  int efd_;
  int monitor_flag_;
  // uac dev
  alsa_device_t* uac_dev_;
  int audio_enable_ = 0;
};
}  // namespace Uvcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // INCLUDE_UVCPLUGIN_UVCPLUGIN_H_
