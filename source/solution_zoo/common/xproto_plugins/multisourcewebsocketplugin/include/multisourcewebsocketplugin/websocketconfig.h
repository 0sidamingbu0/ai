/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_WEBSOCKETCONFIG_H_
#define INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_WEBSOCKETCONFIG_H_

#include <string>
#include <mutex>
#include <memory>
#include <vector>
#include "json/json.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {
class WebsocketPlugin;

class WebsocketConfig {
  friend class WebsocketPlugin;

 public:
  /* type of video frame */
  enum VideoType {
    VIDEO_YUV,
    VIDEO_H264,
    VIDEO_JPG
  };

  /* type of display mode */
  enum DisplayType {
      QT_MODE,
      WEB_MODE
  };

 public:
  WebsocketConfig() = delete;
  explicit WebsocketConfig(const std::string &path);
  bool LoadConfig();
  std::string GetValue(const std::string &key);
  Json::Value GetJson() const;

 private:
  bool CheckConfig();
  std::string path_;
  Json::Value json_;
  std::mutex mutex_;

  uint8_t auth_mode_;
  DisplayType display_mode_;
  uint8_t layer_;
  uint8_t jpeg_quality_;
  std::string user_, password_;
  VideoType video_type_;
  uint32_t image_width_, image_height_;
  uint32_t data_buf_size_, packet_size_;
  std::string attr_des_file_;
  int frame_buf_depth_ = 0;
  int dump_jpg_num_ = 0;
  int jpg_encode_time_ = 0;
  int use_vb_ = 0;
  int is_cbr_ = 1;
  int bitrate_ = 2000;
  std::vector<int> chnn_list_;
};

}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_WEBSOCKETCONFIG_H_
