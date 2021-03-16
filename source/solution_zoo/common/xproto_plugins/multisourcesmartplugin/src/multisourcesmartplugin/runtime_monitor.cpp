/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include "multisourcesmartplugin/runtime_monitor.h"

#include <memory>
#include <mutex>

#include "hobotlog/hobotlog.hpp"
#include "multisourcesmartplugin/util.h"
#include "smartplugin/utils/time_helper.h"
#include "xproto_msgtype/gdcplugin_data.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {
using ImageFramePtr = std::shared_ptr<hobot::vision::ImageFrame>;
using XStreamImageFramePtr = xstream::XStreamData<ImageFramePtr>;
using horizon::vision::xproto::basic_msgtype::IpmImageMessage;

void RuntimeMonitor::PushFrame(const SmartInput *input, int image_idx) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  HOBOT_CHECK(input) << "Null HorizonVisionImageFrame";
  auto frame_info = input->frame_info;
  HOBOT_CHECK(frame_info->num_ > 0);

  uint64_t frame_id;
  uint32_t channel_id;
  if (input->type == "IpmImage") {
    auto ipm_info = std::dynamic_pointer_cast<IpmImageMessage>(frame_info);
    std::shared_ptr<hobot::vision::CVImageFrame> image0 =
      ipm_info->ipm_imgs_[0];
    frame_id = image0->frame_id;
    channel_id = image0->channel_id;
  } else {
    std::shared_ptr<hobot::vision::PymImageFrame> image0 =
        frame_info->image_[image_idx];
    frame_id = image0->frame_id;
    channel_id = image0->channel_id;
  }
  // 查找source id对应的那一组输入的frame
  auto &frame_set_ = input_frames_[channel_id];

  LOGD << "Push source id: " << channel_id
    << ", frame id: " << frame_id;

  if (frame_set_[frame_id].ref_count == 0) {
    LOGD << "Insert frame id:" << frame_id;
    frame_set_[frame_id].image_num = frame_info->num_;
    //  frame_set_[frame_id].img = frame_info->image_;
    frame_set_[frame_id].context = input->context;  // input->context = input
  } else {
    delete input;
  }

  frame_set_[frame_id].ref_count++;
}

RuntimeMonitor::InputFrameData RuntimeMonitor::PopFrame(
    const int32_t &source_id, const int32_t &frame_id) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  InputFrameData input;

  LOGD << "Pop frame source:" << source_id
    << ", frame id:" << frame_id;

  auto frame_set_itr = input_frames_.find(source_id);
  if (frame_set_itr == input_frames_.end()) {
    LOGW << "Unknow source id: " << source_id;
    return input;
  }
  auto &frame_set_ = frame_set_itr->second;

  auto frame_iter = frame_set_.find(frame_id);
  if (frame_iter == frame_set_.end()) {
    LOGW << "Unknow source id: " << source_id
      << ", frame id: " << frame_id;
    return input;
  }
  auto origin_input = frame_iter->second;

  if ((--origin_input.ref_count) == 0) {
    LOGI << "erase frame id:" << frame_id;
    frame_set_.erase(frame_iter);
  }
  return origin_input;
}

void RuntimeMonitor::FrameStatistic() {
    // 实际智能帧率计算
  static int fps = 0;
  // 耗时统计，ms
  static auto lastTime = hobot::Timer::tic();
  static int frameCount = 0;

  ++frameCount;

  auto curTime = hobot::Timer::toc(lastTime);
  // 统计数据发送帧率
  if (curTime > 1000) {
    fps = frameCount;
    frameCount = 0;
    lastTime = hobot::Timer::tic();
    LOGW << "Smart fps = " << fps;
    frame_fps_ = fps;
  }
}

RuntimeMonitor::RuntimeMonitor() { Reset(); }

bool RuntimeMonitor::Reset() { return true; }

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  //  namespace vision
}  //  namespace horizon
