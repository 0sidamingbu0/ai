/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Fei cheng
 * @Mail: fei.cheng@horizon.ai
 * @Date: 2019-09-14 20:38:52
 * @Version: v0.0.1
 * @Brief: mcplugin impl based on xpp.
 * @Last Modified by: Fei cheng
 * @Last Modified time: 2019-09-14 22:41:30
 */

#ifndef APP_INCLUDE_PLUGIN_MCPLUGIN_MCPLUGIN_H_
#define APP_INCLUDE_PLUGIN_MCPLUGIN_MCPLUGIN_H_

#include <future>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "xproto_msgtype/smartplugin_data.h"
#include "xproto_msgtype/vioplugin_data.h"
#include "mcplugin/mcmessage.h"
#include "hobotlog/hobotlog.hpp"
#include "utils/jsonConfigWrapper.h"
#include "xproto/utils/singleton.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
#include "xproto_msgtype/uvcplugin_data.h"
#include "xproto/plugin/xpluginasync.h"
#include "utils/votmodule.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace mcplugin {
using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using horizon::vision::xproto::basic_msgtype::UvcMessage;
using horizon::vision::xproto::basic_msgtype::TransportMessage;
using horizon::vision::xproto::basic_msgtype::SmartMessage;
using horizon::vision::xproto::basic_msgtype::VioMessage;
using horizon::vision::xproto::basic_msgtype::APImageMessage;
using MCMsgPtr = std::shared_ptr<MCMessage>;

class PluginContext : public hobot::CSingleton<PluginContext> {
 public:
  PluginContext() : exit(false) { plugins.clear(); }
  ~PluginContext() = default;

 public:
  bool exit;
  uint32_t basic_plugin_cnt = 3;
  std::vector<std::shared_ptr<XPluginAsync>> plugins;
};

class MCPlugin : public XPluginAsync {
 public:
  MCPlugin() = default;
  explicit MCPlugin(const std::string& config_file);

  ~MCPlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;

 private:
  int StartPlugin(uint64 msg_id);
  int StopPlugin(uint64 msg_id);
  int OnGetUvcResult(const XProtoMessagePtr& msg);
  int OnGetSmarterResult(const XProtoMessagePtr& msg);
  int OnGetVioResult(const XProtoMessagePtr& msg);
  void ParseConfig();
  int GetCPStatus();
  int GetCPLogLevel();
  std::shared_ptr<HorizonVisionImageFrame>
  ConstructVotImgData(const std::shared_ptr<VioMessage>&);
  std::shared_ptr<std::string>
  ConstructVotSmartData(const std::shared_ptr<SmartMessage>&);

 private:
  enum cp_status {
    CP_READY,
    CP_STARTING,
    CP_WORKING,
    CP_STOPPING,
    CP_ABNORMAL,
    CP_UPDATING,
  };
  std::atomic<bool> is_running_;
  std::string config_file_;
  std::shared_ptr<JsonConfigWrapper> config_;
  cp_status cp_status_;
  int log_level_;

  void ConstructMsgForCmd(const x3::InfoMessage& InfoMsg, uint64 msg_id = 0);
  // std::string profile_log_file_;
  std::map<ConfigMessageType, std::function<int(uint64)>> cmd_func_map = {
          {SET_APP_START, std::bind(&MCPlugin::StartPlugin, this,
                                    std::placeholders::_1)},
          {SET_APP_STOP, std::bind(&MCPlugin::StopPlugin, this,
                                   std::placeholders::_1)} };

  bool auto_start_ = false;
  bool enable_vot_ = false;
  std::map<uint64_t, std::shared_ptr<horizon::vision::VotData_t>>
          cache_vot_data_;
  std::mutex mut_cache_;
  std::condition_variable cv_;
  uint64_t cache_len_limit_ = 10;
  std::vector<std::shared_ptr<std::thread>> feed_vo_thread_;
  int feed_vo_thread_num_ = 1;
  int feed_vo_pym_layer_ = -1;
  uint16_t feed_vo_src_w_ = 0;
  uint16_t feed_vo_src_h_ = 0;
  uint16_t feed_vo_dest_w_ = 1920;
  uint16_t feed_vo_dest_h_ = 1080;

  std::shared_ptr<std::thread> status_monitor_thread_ = nullptr;
  std::atomic_int unrecv_ap_count_;
  int unrecv_ap_count_max = 5;
  int OnGetAPImage(const x3::Image &image, uint64_t seq_id);

 private:
  struct gesture_dump_t {
    std::stringstream ss_hand;
    int gest_raw = -1;
    float gest_score = 0;
    int gest_vote = -1;
    int gest_final = -1;
    std::vector<float> model_out_values;
  };
  bool enable_dump_smart_ = false;
  bool enable_append_smart_ = false;
  std::mutex dump_smart_mutex_;
  std::ofstream ofs_dump_smart_;
  bool enable_dump_img_ = false;
  std::string save_dump_img_path_ = "./";
  std::shared_ptr<std::string>
  DumpSmartData(const std::shared_ptr<SmartMessage>&,
                const std::shared_ptr<std::string>&);

  std::shared_ptr<std::thread> task_feedback_ = nullptr;

  // dump action 静态手势:
  enum class dump_gesture_action {
    GESTURE_UNKNOWN = -1,  // 无手势或定义之外的手势
    GESTURE_ONE = 1,  // 食指伸出 1
    GESTURE_ZERO = 2,  // 握拳 2
    GESTURE_LOVE = 3,  //比心 3
    GESTURE_SIX = 4,  // six 4
    GESTURE_FOUR,  // 除大拇指外四指伸出 5
    GESTURE_SHOOT,  // 手枪/八 6
    GESTURE_SILENT,  // 食指放在嘴唇上,其他手指握拳 7
    GESTURE_FIVE,  // 五指伸出 8
    GESTURE_TWO,  // “V” 9
    GESTURE_OK,  // ok 10
    GESTURE_THUMB_LEFT,  // 大拇指向左,其他手指握拳 11
    GESTURE_THUMB_RIGHT,  // 大拇指向右,其他手指握拳 12
    GESTURE_THUMB_UP,  // 大拇指向上,其他手指握拳 13
    GESTURE_THUMB_DOWN  // 大拇指向下,其他手指握拳 14
  };

  std::map<int, dump_gesture_action> horizon_gesture_dump_router_ {
          {0, dump_gesture_action::GESTURE_UNKNOWN},
          {1, dump_gesture_action::GESTURE_LOVE},
          {2, dump_gesture_action::GESTURE_THUMB_UP},
          {3, dump_gesture_action::GESTURE_TWO},
          {4, dump_gesture_action::GESTURE_SILENT},
          {5, dump_gesture_action::GESTURE_FIVE},
          {10, dump_gesture_action::GESTURE_FIVE}
  };

  typedef struct {
    int32_t id = 0;
    hobot::vision::BBox box;
    hobot::vision::Landmarks lmk;
    int age = 0;
    int gender = 0;
    int dist = 0;
  } smart_data_face_t;
  typedef struct {
    int32_t id = 0;
    hobot::vision::BBox box;
    hobot::vision::Landmarks lmk;
  } smart_data_body_t;
  typedef struct {
    int32_t id = 0;
    hobot::vision::BBox box;
    hobot::vision::Landmarks lmk;
    int gesture = -1;
    float gesture_score = 0;
  } smart_data_hand_t;

  typedef struct {
    std::unordered_map<int32_t, smart_data_face_t> faces;
    std::unordered_map<int32_t, smart_data_body_t> bodys;
    std::unordered_map<int32_t, smart_data_hand_t> hands;
  } smart_data_t;
  bool enable_dump_smart_2_json_ = false;
  std::mutex dump_smart_2_json_mutex_;
  Json::Value root_;
  std::unordered_map<std::string, std::string> map_feedback_namelist_;
  void DumpSmart2Json(const std::shared_ptr<SmartMessage>&,
                      const std::shared_ptr<std::string>&);
};

// int MCPlugin::cp_status_;
// int MCPlugin::log_level_;

}  // namespace mcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // APP_INCLUDE_PLUGIN_MCPLUGIN_MCPLUGIN_H_
