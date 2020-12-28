/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2019 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef APP_INCLUDE_PLUGIN_SMARTPLUGIN_SMARTMESSAGE_H_
#define APP_INCLUDE_PLUGIN_SMARTPLUGIN_SMARTMESSAGE_H_
#include <memory>
#include <string>
#include <vector>

#include "xwarehouse/xwarehouse.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/plugin/xpluginasync.h"
#include "hobotxsdk/xstream_data.h"
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin_multiplebox {

using Box = hobot::vision::BBox;
using Point = hobot::vision::Point;

/**
 * @brief timestamp, freq id
 */
struct UserData {
  int64_t frame_id_;
  int64_t timestamp_;
  Box box_;
};


/**
 * @brief float array : use vision_type.hpp
 */
struct FloatArray : public hobot::vision::FloatArray {
  std::string type_;
};

struct FeatureInfo {
  std::string type_ = "person";
  int64_t frame_id_;
  int64_t time_stamp_;
  uint32_t track_id_;
  std::vector<FloatArray> float_arrays_;  // 每个人多个抓拍特征
};

struct FeatureFrameMessage {
  //  消息错误码，指明此帧数据状态
  //  0：正常数据，1：capture drop数据，2:feature drop数据
  uint32_t error_code_;
  int ch_id_;
  std::vector<FeatureInfo> feature_infos_;  // 每帧多个人
};

struct RecogResult_t {
  uint64_t ch_id;
  uint64_t track_id;
  uint32_t is_recognize;
  float similar;
  std::string record_id;
  std::string img_uri_list;
};


}   //  namespace smartplugin_multiplebox
}   //  namespace xproto
}   //  namespace vision
}   //  namespace horizon
#endif  // APP_INCLUDE_PLUGIN_SMARTPLUGIN_SMARTMESSAGE_H_
