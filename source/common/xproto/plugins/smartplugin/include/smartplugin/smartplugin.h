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

#ifndef INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_
#define INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "xproto/msgtype/include/xproto_msgtype/protobuf/x3.pb.h"

#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"

#include "hobotxsdk/xstream_sdk.h"
#include "smartplugin/runtime_monitor.h"
#include "smartplugin/smart_config.h"
#include "smartplugin/traffic_info.h"
#include "xproto_msgtype/smartplugin_data.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;

using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;
using horizon::vision::xproto::basic_msgtype::SmartMessage;

struct VehicleSmartMessage : SmartMessage {
 public:
  VehicleSmartMessage();
  std::string Serialize() override;
  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h);
  void Serialize_Print(Json::Value &root);
  void Serialize_Dump_Result();
 public:
  int camera_type;
  std::vector<VehicleInfo> vehicle_infos;
  std::vector<PersonInfo> person_infos;
  std::vector<NoMotorVehicleInfo> nomotor_infos;
  std::vector<VehicleCapture> capture_infos;
  std::vector<uint64_t> lost_track_ids;
  std::vector<AnomalyInfo> anomaly_infos;

  std::vector<TrafficConditionInfo> traffic_condition_infos;
  TrafficStatisticsInfo traffic_statistics_info;
};

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

struct CustomSmartMessage : SmartMessage {
  explicit CustomSmartMessage(
    xstream::OutputDataPtr out) : smart_result(out) {
    type_ = TYPE_SMART_MESSAGE;
  }
  std::string Serialize() override;
  std::string Serialize(int ori_w, int ori_h, int dst_w, int dst_h) override;
  void Serialize_Print(Json::Value &root);
  void SetAPMode(bool ap_mode) {
    ap_mode_ = ap_mode;
  }
  void Serialize_Dump_Result();

 protected:
  xstream::OutputDataPtr smart_result;
  bool ap_mode_ = false;

 private:
  enum class gesture_type {
    Background,
    FingerHeart,
    ThumbUp,
    Victory,
    Mute,  // 4
    PalmMove,  // 5
    IndexFingerRotateAntiClockwise,  // mirror image
    IndexFingerRotateClockwise,
    Pinch,
    Palmpat,  // 9
    // model output end
    // gesture output with strategy start
    Palm
  };

  typedef struct {
    // frame id is used to record the time of gesture
    // type is used to:
    // 1. check if hand id ever has palm/palmmove gesture in lastest N frames
    // 2. calculate conflict gesture num
    // if conflict num exceeds limit, clear the cache of this hand id
    // update frame id and gesture type only when the gesture is
    // Palm/PalmMove/Palmpat/Pinch/Rotate
    // the gesture type maybe from callback
    uint64 frame_id = 0;
    gesture_type type = gesture_type::Background;
  } gesture_single_frame_info_t;
  typedef struct {
    // the size of frame_infos and points are same
    // the gesture and frame id are stored in frame_infos
    // the gesture point is stored in points,
    // which can be used for fitting with openCV
    std::vector<gesture_single_frame_info_t> frame_infos;
    std::vector<cv::Point> points;

    // if gesture is rotate, rotate_num, angel and fit results are valid
    int rotate_num = 0;
    int angel = 0;
    float fit_ratio = 0.0;
    int fit_residual_err = 0;
    cv::RotatedRect fit_rect;
    // has_start notes whether a new rotate round is start
    bool has_start = false;

    // if present gesture is not same with first, conflict num will increase
    // the gesture cache will be clear if conflict num exceeds limit
    // gesture conflict is allowed in order to
    // enhance the continuation of dynamic gesture
    int gesture_conflict_num = 0;

    // last frame id is used for gesture cache clear
    // last type is used for gesture callback
    // update frame id and gesture type only when the gesture is
    // Palm/PalmMove/Palmpat/Pinch/Rotate and the gesture is not from callback
    uint64 last_frame_id = 0;
    gesture_type last_type = gesture_type::Background;
  } gesture_info_t;

  struct EllipsePara {
    cv::Point2f c;
    float A;
    float B;
    float C;
    float F;
  };

  static std::map<int, int> fall_state_;
  static std::mutex static_attr_mutex_;
  static std::mutex fall_mutex_;
  static std::map<int, int> gesture_state_;
  static std::map<int, float> gesture_start_time_;
  static std::map<int, gesture_info_t> gesture_info_cache_;

 private:
  // cal the angel of lines pt1-c and pt2-c using pt1 as base
  float CalAngelOfTwoVector(const cv::Point2f &c,
                            const cv::Point2f &pt1,
                            const cv::Point2f &pt2,
                            bool clock_wise = true);
  // cal the ellipse para using opencv fitting RotatedRect
  EllipsePara CalEllipsePara(const cv::RotatedRect& fit_rect);
  // cal the fitting point of y val using the original point of x and y
  float CalFitVal(const cv::RotatedRect& fit_rect,
                  float x_origin, float y_origin);
  int UpdateGestureInfo(const gesture_type& gesture_val,
                        const gesture_type& gesture_original,
                        int hand_id,
                        const hobot::vision::Point& hand_lmk_point,
                        const std::shared_ptr<xstream::XStreamData<
                                hobot::vision::BBox>>& hand_box,
                        float y_ratio);
};

class SmartPlugin : public XPluginAsync {
 public:
  SmartPlugin() = default;
  explicit SmartPlugin(const std::string& config_file);

  void SetConfig(const std::string& config_file) { config_file_ = config_file; }

  ~SmartPlugin() = default;
  int Init() override;
  int DeInit() override;
  int Start() override;
  int Stop() override;
  std::string desc() const { return "SmartPlugin"; }

 private:
  // 获取单路图像，workflow配置的图像输入节点名字
  // SmartPlugin派生类可以根据需要修改输入节点的名字
  // 但是必须保证该接口返回的图像输入节点名字和xstream json配置中一致
  virtual std::string GetWorkflowInputImageName() {
    return "image";  // 当前沉淀的solution均使用image这个
  }

  // 创建xproto框架下感知结果的消息对象
  // 感知结果消息对象必须是CustomSmartMessage或者集成自CustomSmartMessage
  // 输入参数xstream_out为xstream workflow执行完成，xstream回调返回的数据对象
  virtual std::shared_ptr<CustomSmartMessage>
  CreateSmartMessage(xstream::OutputDataPtr xstream_out) {
    // 当前沉淀的解决方案，默认为CustomSmartMessage对象
    return std::make_shared<CustomSmartMessage>(xstream_out);
  }

  int Feed(XProtoMessagePtr msg);
  void OnCallback(xstream::OutputDataPtr out);
  void ParseConfig();

  std::shared_ptr<XStreamSDK> sdk_;
  int sdk_monitor_interval_;
  std::string config_file_;
  std::shared_ptr<RuntimeMonitor> monitor_;
  std::shared_ptr<JsonConfigWrapper> config_;
  std::string xstream_workflow_cfg_file_;
  bool enable_profile_{false};
  std::string profile_log_file_;
  bool result_to_json_{false};
  bool dump_result_{false};
  Json::Value root_;
  bool run_flag_ = false;

#ifdef USE_MC
  int OnApInfoMessage(const XProtoMessagePtr &msg);
#endif
};



}  // namespace smartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_SMARTPLUGIN_SMARTPLUGIN_H_
