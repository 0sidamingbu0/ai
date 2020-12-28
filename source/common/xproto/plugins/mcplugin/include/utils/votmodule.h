/*
 * @Description:
 * @Author: xx@horizon.ai
 * @Date: 2020-06-22 16:17:25
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#ifndef INCLUDE_VOTMODULE_H_
#define INCLUDE_VOTMODULE_H_
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include "hb_vio_interface.h"
#include "hb_vot.h"
#include "hobot_vision/blocking_queue.hpp"
#include "horizon/vision_type/vision_msg.h"

namespace horizon {
namespace vision {

struct RecogResult_t {
  uint64_t track_id;
  uint32_t is_recognize;
  float similar;
  std::string record_id;
  std::string img_uri_list;
};

struct VotData_t {
  // only supports pym0
  bool is_drop_frame_ = false;
  std::shared_ptr<HorizonVisionImageFrame> sp_img_ = nullptr;
  std::shared_ptr<RecogResult_t> sp_recog_ = nullptr;
  std::shared_ptr<std::string> sp_smart_pb_ = nullptr;
};

struct VotDrawCtrl_t {
  bool draw_hand_lmk = true;
  bool draw_face_box = true;
  bool draw_head_box = false;
  bool draw_body_box = false;
  bool draw_face_lmk = false;
  bool draw_age_gender_dist = true;
  bool draw_body_kps = false;
  bool draw_id = false;
};

class VotModule
{
 public:
  static std::shared_ptr<VotModule>& Instance() {
    static std::shared_ptr<VotModule> vot_module_;
    static std::once_flag init_flag;
    std::call_once(init_flag, [](){
        vot_module_ = std::shared_ptr<VotModule>(new VotModule());
    });
    return vot_module_;
  }

  ~VotModule();
  int Input(const std::shared_ptr<VotData_t>&);
  int Start();
  int Stop();

 private:
  uint32_t group_id_;
  uint32_t channel_ = 0;
  uint32_t image_width_;
  uint32_t image_height_;
  uint32_t image_data_size_;
  // display buffer
  char *buffer_;
  std::atomic_bool stop_;

  VotModule();
  int Init();
  int DeInit();

  void bgr_to_nv12(uint8_t *bgr, char *buf);
  std::shared_ptr<char> PlotImage(const std::shared_ptr<VotData_t>& vot_data);

 private:
  // display buffer
  // key is frame id and is used for sort
  std::map<uint64_t, std::shared_ptr<char>> send_vot_cache_;
  std::mutex send_vot_mtx_;
  std::condition_variable send_vot_cv_;
  uint64_t need_send_vot_id_ = 0;
  std::shared_ptr<std::thread> send_vot_task_ = nullptr;

  std::shared_ptr<std::thread> display_task_ = nullptr;

  // vot module input
  hobot::vision::BlockingQueue<std::shared_ptr<VotData_t>> in_queue_;
  uint32_t in_queue_len_max_ = 10;

  // plot task
  uint32_t plot_task_num_ = 4;
  std::vector<std::shared_ptr<std::thread>> plot_tasks_;

  // recog res cache
  std::unordered_map<uint64_t, std::shared_ptr<RecogResult_t>>
          recog_res_cache_;
  std::mutex cache_mtx_;

  // true: convert nv12 to bgr and plot on bgr img,
  // the convert task is time-cost
  // false: plot on nv12 img directly
  bool plot_color_ = false;
  // true: plot vio drop img, which has no smart data
  // false: do not plot vio drop img
  bool plot_drop_img_ = false;

  int frame_fps_ = -1;

  VotDrawCtrl_t vot_draw_ctrl_;

  enum class output_gesture_type {
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
    Palm
  };

  std::vector<std::string> gesture_type_str {
    "Background",
    "FingerHeart",
    "ThumbUp",
    "Victory",
    "Mute",  // 4
    "PalmMove",
    "AntiClockwise",  // mirror image
    "Clockwise",
    "Pinch",  // 8
    "Palmpat",
    "Palm"
  };
};

}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_VOTMODULE_H_
