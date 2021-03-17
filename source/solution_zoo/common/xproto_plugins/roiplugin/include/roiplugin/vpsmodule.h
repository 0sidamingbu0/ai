/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail: @horizon.ai
 */
#ifndef APP_INCLUDE_PLUGIN_ROIPLUGIN__VPSMODULE_H_
#define APP_INCLUDE_PLUGIN_ROIPLUGIN__VPSMODULE_H_

#include <string.h>

#include <memory>
#include <thread>
#include <vector>

#include "hb_vio_interface.h"
#include "roiplugin/roi_common.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace roiplugin {
class VpsModule {
 public:
  VpsModule();
  ~VpsModule();
  int Init();
  int Start();
  int Stop();
  int DeInit();

 public:
  int Input(std::shared_ptr<VideoRoiData> &in_data);

  int UpdateViewRoi(const RoiInfo &info);
  int UpdateTrackRoi(const RoiInfo &info, const RoiInfo *info_tmp,
                     const bool is_second = false);

  int OutputViewData(std::shared_ptr<VideoRoiData> &out_data);
  int OutputTrackData(std::shared_ptr<VideoRoiData> &out_data,
                      const bool is_second = false);

 public:
  int Process(const RoiInfo &info, std::shared_ptr<VideoRoiData> &data,
              const RoiInfo *info_tmp = nullptr, const bool is_second = false,
              const bool send_video = true);

  int Process(std::shared_ptr<VideoRoiData> &in_data,
              std::shared_ptr<VideoRoiData> &out_data, const RoiInfo &info,
              const RoiInfo *info_tmp = nullptr, const bool is_second = false,
              const bool send_video = true);

 private:
  int GetFrame(const int group, const int channel, hb_vio_buffer_t *gdc_buf,
               std::shared_ptr<VideoRoiData> &out_data);

 private:
  int SysInit();
  int SysUnit();

 private:
  bool init_ = false;
  uint32_t timeout_ = 2000;
  mutable std::mutex mutex_;
  hb_vio_buffer_t *buffer_ = nullptr;
  hb_vio_buffer_t *view_buf = nullptr;
  hb_vio_buffer_t *track1_buf = nullptr;
  hb_vio_buffer_t *track2_buf = nullptr;
};

}  // namespace roiplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // APP_INCLUDE_PLUGIN_ROIPLUGIN__VPSMODULE_H_
