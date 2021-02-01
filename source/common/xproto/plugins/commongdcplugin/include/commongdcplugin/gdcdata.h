/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/

#ifndef APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_DATA_H_
#define APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_DATA_H_
#include <vector>
#include <string>
#include "horizon/vision_type/vision_type.hpp"
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <stdint.h>
#include <string.h>
#include "hb_sys.h"
#include "hb_vio_interface.h"
#include "hb_vps_api.h"
#include "hb_vp_api.h"
#ifdef __cplusplus
}
#endif  /* __cplusplus */
#define GDC_GROUP_ID (4)

namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

using hobot::vision::PymImageFrame;
using hobot::vision::SrcImageFrame;
using hobot::vision::ImageLevelInfo;

struct GdcConfig {
  bool gdc_enable;
  bool all_in_one_vio;
  int data_source_num;
  int pym_layer;
  std::vector<int> channel2direction;
  std::vector<std::string> gdc_file_path;
  ~GdcConfig() {}
};

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_DATA_H_
