/*
 * Copyright (c) 2020-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     hid_manager.h
 * \Author   zhe.sun
 * \Version  1.0.0.0
 * \Date     2020/6/10
 * \Brief    implement of api header
 */
#ifndef INCLUDE_UVCPLUGIN_HIDMANAGER_H_
#define INCLUDE_UVCPLUGIN_HIDMANAGER_H_
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <deque>
#include <vector>
#include <utility>

#include "hobot_vision/blocking_queue.hpp"
#include "./smart_manager.h"
#include "json/json.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto_msgtype/smartplugin_data.h"

using horizon::vision::xproto::basic_msgtype::SmartMessage;
using SmartMessagePtr = std::shared_ptr<SmartMessage>;

namespace horizon {
namespace vision {
namespace xproto {
namespace Uvcplugin {
using horizon::vision::xproto::XProtoMessagePtr;

#define HID_MAX_PACKET_SIZE (1024)
#define HID_BUFFER_SIZE (10*1024*1024)  // 10M bytes

class HidManager : public SmartManager {
 public:
  HidManager() = delete;
  explicit HidManager(std::string config_path);
  ~HidManager();

  virtual int Init();
  virtual int DeInit();
  virtual int Start();
  virtual int Stop();

  virtual int FeedSmart(XProtoMessagePtr msg, int ori_image_width,
                        int ori_image_height, int dst_image_width,
                        int dst_image_height);
  int FeedDropSmart(uint64_t frame_id) override;
  int FeedInfo(const XProtoMessagePtr& msg);

 private:
  int Send(const std::string &proto_str);
  void SendThread();
  void RecvThread();
  int Serialize(SmartMessagePtr smart_msg, int ori_image_width,
                int ori_image_height, int dst_image_width,
                int dst_image_height);
  int SerializeDropFrame(uint64_t frame_id);

 public:
  // AP->CP info buffer
  bool IsApMode() { return ap_mode_;}
  bool SetApMode(bool set) { return ap_mode_ = set;}
  hobot::vision::BlockingQueue<std::string> pb_ap2cp_info_queue_;

 private:
  bool ap_mode_ = false;
  bool stop_flag_;
  std::string config_path_;
  Json::Value config_;
  std::shared_ptr<std::thread> thread_;
  std::shared_ptr<std::thread> recv_thread_;
  enum SmartType { SMART_FACE, SMART_BODY, SMART_VEHICLE };
  SmartType smart_type_ = SMART_BODY;

  std::string hid_file_ = "/dev/hidg0";
  int hid_file_handle_ = -1;
  std::mutex queue_lock_;
  const unsigned int queue_max_size_ = 50;

  using SmartMsg = std::pair<uint64_t, std::string>;
  struct compare{
    bool operator()(SmartMsg msg1, SmartMsg msg2){
      return msg1.first > msg2.first;
    }
  };
  std::priority_queue<SmartMsg,
          std::deque<SmartMsg>, compare> smart_msg_queue_;

  std::condition_variable condition_;

  // CP->AP info buffer
  hobot::vision::BlockingQueue<std::string> pb_cp2ap_info_queue_;

  typedef struct {
    char null_array[HID_MAX_PACKET_SIZE];
  } buffer_offset_size_t;
  hobot::CThreadPool serialize_thread_;
  bool print_timestamp_ = false;
  int Recv(char *recv_data);
};

}  // namespace Uvcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // INCLUDE_UVCPLUGIN_HIDMANAGER_H_
