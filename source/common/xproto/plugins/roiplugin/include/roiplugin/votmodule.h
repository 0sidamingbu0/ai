/*
 * @Description:
 * @Author: xx@horizon.ai
 * @Date: 2020-12-19 16:17:25
 * @Copyright Horizon Robotics, Inc.
 */
#ifndef INCLUDE_UTILS_ROI_VOTMODULE_H_
#define INCLUDE_UTILS_ROI_VOTMODULE_H_
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hb_vio_interface.h"
#include "hb_vot.h"
#include "hobot_vision/blocking_queue.hpp"
#include "horizon/vision_type/vision_msg.h"
#include "roiplugin/roi_common.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace roiplugin {
class VotModule {
 public:
  VotModule();
  ~VotModule();

 public:
  int Init();
  int DeInit();
  int Input(const std::shared_ptr<VideoRoiData> &);
  int Start();
  int Stop();

 private:
  int HandleData();
  void PlotImage(const std::shared_ptr<VideoRoiData> &vot_data);

 private:
  bool init_ = false;
  // display buffer
  char *buffer_;
  std::atomic_bool stop_;

  bool plot_drop_img_ = false;
  int frame_fps_ = 0;

  // vot module input
  hobot::vision::BlockingQueue<std::shared_ptr<VideoRoiData>> in_queue_;
  uint32_t in_queue_len_max_ = 40;
  std::thread plot_task_;

  // std::condition_variable condition_;
  // std::vector<std::map<uint64_t, std::shared_ptr<VideoRoiData>>> video_list_;
  // std::mutex mut_cache_;
};

}  // namespace roiplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_UTILS_ROI_VOTMODULE_H_
