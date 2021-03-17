/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail: @horizon.ai
 */

#ifndef INCLUDE_RTSPPLUGIN_SMARTPLUGIN_H_
#define INCLUDE_RTSPPLUGIN_SMARTPLUGIN_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hobotxsdk/xstream_sdk.h"
#include "horizon/vision_type/vision_type.h"
#include "json/json.h"
#include "mediapipemanager/meidapipeline.h"
#include "rtspclient/rtspclient.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto_msgtype/rtspplugin_data.h"

#ifdef __cplusplus
extern "C" {
#include "hb_comm_vdec.h"
#include "hb_comm_venc.h"
#include "hb_comm_video.h"
#include "hb_common.h"
#include "hb_sys.h"
#include "hb_type.h"
#include "hb_vdec.h"
#include "hb_venc.h"
#include "hb_vio_interface.h"
#include "hb_vp_api.h"
#include "hb_vps_api.h"
}
#endif

namespace horizon {
namespace vision {
namespace xproto {
namespace rtspplugin {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;

using horizon::vision::xproto::basic_msgtype::RtspMessage;
using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;

class RtspPlugin : public XPluginAsync {
 public:
  RtspPlugin() = default;
  explicit RtspPlugin(const std::string &config_file) {
    config_file_ = config_file;
  }

  void SetConfig(const std::string &config_file) { config_file_ = config_file; }

  ~RtspPlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;
  void GetDecodeFrame0();
  void GetDeocdeFrame(std::shared_ptr<MediaPipeLine> pipeline, int channel);
  void WaitToStart();
  void GetConfigFromFile(const std::string &path);

 private:
  int Feed(XProtoMessagePtr msg);
  void OnCallback(xstream::OutputDataPtr out);
  int DecodeInit();
  void Process();
  void CheckRtspState();
  void ProcessData(const int channel_id, pym_buffer_t *out_pym_buf);

  void ParseConfig();
  void GetRtspConfigFromFile(const std::string &path);
  void GetDropFrameConfigFromFile(const std::string &path);
  void GetDisplayConfigFromFile(const std::string &path);

  std::vector<std::thread> threads_;
  std::vector<ourRTSPClient *> rtsp_clients_;

  struct Rtspinfo {
    std::string url;
    bool tcp_flag;
    int frame_max_size;
    bool save_stream;
  };
  std::vector<Rtspinfo> rtsp_url_;

  std::vector<bool> rtsp_clients_stat_;
  int image_width_;
  int image_height_;
  int channel_number_;
  PAYLOAD_TYPE_E codec_type_;
  std::string config_file_;
  Json::Value config_;
  std::shared_ptr<std::thread> process_thread_;
  bool running_;
  static int frame_count_;
  static std::mutex framecnt_mtx_;

  std::shared_ptr<std::thread> check_thread_;
  TaskScheduler *scheduler_;
  UsageEnvironment *env_;

  bool drop_frame_ = false;
  int drop_frame_interval_ = 0;

  bool not_run_smart_ = false;
  bool running_venc_1080p_ = false;
  bool running_venc_720p_ = false;
  bool running_vot_ = true;
  bool encode_smart_ = false;
  bool draw_smart_ = false;
  int display_mode_ = 0;
  bool transition_support_ = false;
  bool draw_real_time_video_ = true;

  int buffer_count_ = 4;
  std::vector<pym_buffer_t *> pym_buffers_;
  std::vector<bool> use_flag_list;
  std::mutex pym_buffer_mutex_;
};

}  // namespace rtspplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_
