/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCESMARTPLUGIN_RUNTIME_MONITOR_H_
#define INCLUDE_MULTISOURCESMARTPLUGIN_RUNTIME_MONITOR_H_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>

#include "hobotxsdk/xstream_data.h"
#include "horizon/vision_type/vision_type.h"
#include "xproto_msgtype/vioplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {
using horizon::vision::xproto::basic_msgtype::VioMessage;

struct SmartInput {
  std::shared_ptr<VioMessage> frame_info;
  void *context;
  std::string type;
};

class RuntimeMonitor {
 public:
  RuntimeMonitor();

  struct InputFrameData {
    uint32_t image_num;
    //  std::vector<std::shared_ptr<hobot::vision::PymImageFrame>> img;
    void *context = nullptr;
    int ref_count = 0;
  };
  bool Reset();

  void PushFrame(const SmartInput *input, int image_idx = 0);

  InputFrameData PopFrame(const int &source_id, const int &frame_id);

  void FrameStatistic();
  int GetFrameFps() { return frame_fps_; }

 private:
  // <source id, frame id> => frame
  std::unordered_map<int32_t, std::unordered_map<int32_t, InputFrameData>>
      input_frames_;
  std::mutex map_mutex_;
  int frame_fps_ = 0;
};

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_MULTISOURCESMARTPLUGIN_RUNTIME_MONITOR_H_
