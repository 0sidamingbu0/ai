/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail: @horizon.ai
 */
#ifndef INCLUDE_SMARTPLUGIN_BOX_PLOTSMARTDATA_H_
#define INCLUDE_SMARTPLUGIN_BOX_PLOTSMARTDATA_H_

#include <string.h>

#include <memory>
#include <mutex>
#include <vector>

#include "video_box_common.h"

namespace horizon {
namespace vision {
class PlotSmartData {
 public:
  PlotSmartData();
  ~PlotSmartData();

 public:
  int PlotData(const smart_vo_cfg_t vo_confg,
               std::shared_ptr<VideoData> video_data,
               HorizonVisionSmartFrame *smart_frame, const bool face = true,
               const bool head = true, const bool body = true,
               const bool kps = true, const bool veh = true);

 private:
  void bgr_to_nv12(uint8_t *bgr, uint8_t *buf, const uint32_t width,
                   const uint32_t height);
};

}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_SMARTPLUGIN_BOX_PLOTSMARTDATA_H_
