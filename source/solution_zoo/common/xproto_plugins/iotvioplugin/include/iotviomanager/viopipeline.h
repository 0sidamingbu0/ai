/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-08-12
 * @Version: v1.0.0
 * @Brief: IOT VIO PipeLine for Horizon VIO System.
 */
#ifndef INCLUDE_VIO_PIPELINE_H_
#define INCLUDE_VIO_PIPELINE_H_
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "horizon/vision_type/vision_type.hpp"
#include "iotviomanager/vio_data_type.h"
#include "iotviomanager/vinmodule.h"
#include "iotviomanager/vpsmodule.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

using hobot::vision::PymImageFrame;

class VioPipeLine {
 public:
  VioPipeLine() = delete;
  VioPipeLine(const std::vector<std::string> &pipe_file, const int &pipe_id)
    : pipe_file_(pipe_file), pipe_start_idx_(pipe_id) {}
  ~VioPipeLine() {}

  int Init();
  int DeInit();
  int Start();
  int Stop();
  int GetPipeId() { return pipe_start_idx_; }
  /**
   * single pipeline info
   * 1.GetInfo: GetPymInfo, GetIpuInfo, GetFbSrcInfo
   * 2.SetInfo: SetFbPymProcess
   * 3.FreeInfo: FreeIpuInfo, FreePymInfo, FreeFeedbackSrcInfo
   *
   *  info_type is defined in vio_data_type.h
   */
  int GetInfo(uint32_t info_type, void *data);
  int SetInfo(uint32_t info_type, void *data);
  int FreeInfo(uint32_t info_type, void *data);
#if 0
  int GetPymInfo(pym_buffer_t *pym_info);
  int GetIpuInfo(hb_vio_buffer_t *ipu_info);
  int GetFbSrcInfo(hb_vio_buffer_t *fb_info);
  int SetFbPymProcess(hb_vio_buffer_t *src_info);
  int FreeIpuInfo(hb_vio_buffer_t *ipu_info);
  int FreePymInfo(pym_buffer_t *pym_info);
  int FreeFeedbackSrcInfo(hb_vio_buffer_t *feed_info);
#endif
  int GetMultiPymInfo(std::vector<std::shared_ptr<PymImageFrame>> &pym_images);
  int FreeMultiPymInfo(std::vector<std::shared_ptr<PymImageFrame>> &pym_images);
  int GetMultiFbSrcInfo(
      std::vector<std::shared_ptr<SrcImageFrame>> &src_images);
  int SetMultiFbPymInfo(
      std::vector<std::shared_ptr<SrcImageFrame>> &src_images);

 private:
  int HbPipeConfig();
  int HbPipeSysBind(int pipe_id);

 private:
  std::vector<std::string> pipe_file_;
  int pipe_start_idx_;
  int pipe_sync_num_ = 0;
  int cam_en_;
  IotVioCfg    vio_cfg_[MAX_PIPE_NUM] = { 0 };
  IotVinParams vin_params_[MAX_PIPE_NUM] = { 0 };
  std::vector<std::pair<int, std::shared_ptr<VinModule>>> vin_module_list_;
  std::vector<std::pair<int, std::shared_ptr<VpsModule>>> vps_module_list_;
};

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_VIO_PIPELINE_H_
