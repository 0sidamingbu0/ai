/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/

#ifndef APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_MANAGER_H_
#define APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_MANAGER_H_
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "commongdcplugin/gdcdata.h"
#include "commongdcplugin/gdcmodule.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

class GdcManager {
 public:
  static std::shared_ptr<GdcManager>& Instance() {
    static std::shared_ptr<GdcManager> gdc_manager;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      gdc_manager = std::shared_ptr<GdcManager>(new GdcManager());
    });
    return gdc_manager;
  }

  GdcManager() = default;
  ~GdcManager() = default;
  void SetConfig(std::shared_ptr<GdcConfig> &config) { gdc_cfg_ = config; }
  int Init();
  int DeInit();
  int Start();
  int Stop();
  int Input(std::shared_ptr<PymImageFrame> &pym_image);
  int MultiInput(std::vector<std::shared_ptr<PymImageFrame>> &pym_images);
  int Output(std::shared_ptr<PymImageFrame> &pym_image,
      std::shared_ptr<SrcImageFrame> &gdc_image);
  int MultiOutput(std::vector<std::shared_ptr<PymImageFrame>> &pym_images,
      std::vector<std::shared_ptr<SrcImageFrame>> &gdc_images);
  int FreeSrcImage(int chn_id,
      std::vector<std::shared_ptr<SrcImageFrame>> &gdc_images);

 private:
  std::shared_ptr<GdcConfig> gdc_cfg_;
  bool gdc_enable_ = false;
  bool is_running_ = false;
  std::shared_ptr<GdcModule> gdc_module_;
};

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_MANAGER_H_
