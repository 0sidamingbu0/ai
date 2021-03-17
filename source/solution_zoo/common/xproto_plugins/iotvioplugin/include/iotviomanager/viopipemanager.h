/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-08-12
 * @Version: v1.0.0
 * @Brief: IOT VIO Pipe Manager for Horizon VIO System.
 */
#ifndef INCLUDE_VIO_PIPE_MANAGER_H_
#define INCLUDE_VIO_PIPE_MANAGER_H_
#include <string>
#include <vector>

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

class VioPipeManager {
 public:
  static VioPipeManager &Get() {
    static VioPipeManager inst;
    return inst;
  }

  int GetPipeId(const std::vector<std::string> &cfg_file);
  int VbInit();
  int VbDeInit();
  /* alloc ion memory include two parts, y and uv buf(such as nv12 format) */
  int AllocVbBuf2Lane(int index, void *buf, uint32_t size_y,
      uint32_t size_uv, int vb_cache_en);
  /* free ion memory include two parts, y and uv buf(such as nv12 format) */
  int FreeVbBuf2Lane(int index, void *buf);
  /**
   * need run reset interface before every unittest init in single process,
   * which including multiple unit test case
   */
  int Reset();

 private:
  std::mutex mutex_;
  int pipe_id_ = 0;
  int vb_init_ref_cnt_ = 0;
  bool vb_init_ = false;
};

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_VIO_PIPE_MANAGER_H_
