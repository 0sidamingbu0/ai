/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_WEBSOCKETPLUGIN_H_
#define INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_WEBSOCKETPLUGIN_H_
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <map>

#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto/threads/threadpool.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
#include "xproto_msgtype/smartplugin_data.h"
#include "multisourcewebsocketplugin/uws_server.h"
#include "multisourcewebsocketplugin/websocketconfig.h"
#include "media_codec/media_codec_manager.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessagePtr;
using horizon::vision::xproto::basic_msgtype::SmartMessagePtr;

struct CompareFrame {
  bool operator()(const x3::FrameMessage &f1, const x3::FrameMessage &f2) {
    return (f1.timestamp_() > f2.timestamp_());
  }
};

struct CompareMsg {
  bool operator()(const SmartMessagePtr m1, const SmartMessagePtr m2) {
    return (m1->time_stamp > m2->time_stamp);
  }
};

class WebsocketPlugin : public xproto::XPluginAsync {
 public:
  WebsocketPlugin() = delete;
  explicit WebsocketPlugin(std::string config_path);
  ~WebsocketPlugin() override;
  int Init() override;
  int Start() override;
  int Stop() override;
  std::string desc() const { return "MultiWebsocketPlugin"; }

 private:
  // 这里需要与SmartPlugin产生的感知消息匹配
  // 一般情况，若SmartPlugin的派生类产生新的消息类型
  // (继承CustomSmartMessage),则需要继承WebsocketPlugin，
  // 重写GetSmartMessageType接口
  virtual std::string GetSmartMessageType() {
    // 当前解决方案默认使用TYPE_SMART_MESSAGE
    return TYPE_SMART_MESSAGE;
  }
  int FeedVideo(XProtoMessagePtr msg);
  int FeedSmart(XProtoMessagePtr msg);
  int SendSmartMessage(SmartMessagePtr msg, x3::FrameMessage &fm);

  void EncodeJpg(XProtoMessagePtr msg, uint32_t image_idx);
  void ParseConfig();
  int Reset();
  void MapSmartProc();
  int SaveYuvToFile(int y_len, int uv_len, uint8_t *y_vaddr, uint8_t *uv_vaddr,
               std::string name);

 private:
  std::map<int, std::shared_ptr<UwsServer>> uws_server_;
  // channel_id, encode_channel_id
  std::map<int, int> video_encode_chn_;
  std::string config_file_;
  std::shared_ptr<WebsocketConfig> config_;
  std::shared_ptr<std::thread> worker_;
  std::mutex map_smart_mutex_;
  bool map_stop_ = false;
  std::condition_variable map_smart_condition_;
  const uint8_t cache_size_ = 25;  // max input cache size
  // channel_id, frame;
  std::map<int,
           std::priority_queue<x3::FrameMessage, std::vector<x3::FrameMessage>,
                               CompareFrame>>
      x3_frames_;
  // channel_id, smart_result
  std::map<int, std::priority_queue<SmartMessagePtr,
                                    std::vector<SmartMessagePtr>, CompareMsg>>
      x3_smart_msg_;
  int origin_image_width_ = 1920;  // update by FeedVideo
  int origin_image_height_ = 1080;
  int dst_image_width_ = 1920;  // update by FeedVideo
  int dst_image_height_ = 1080;
  std::mutex smart_mutex_;
  bool smart_stop_flag_;
  std::mutex video_mutex_;
  bool video_stop_flag_;
  hobot::CThreadPool jpg_encode_thread_;
  hobot::CThreadPool data_send_thread_;
};

}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_WEBSOCKETPLUGIN_H_
