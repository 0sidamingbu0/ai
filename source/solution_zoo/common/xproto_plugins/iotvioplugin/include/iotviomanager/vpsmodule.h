/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef INCLUDE_VPS_MODULE_H_
#define INCLUDE_VPS_MODULE_H_
#include <semaphore.h>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include "horizon/vision_type/vision_type.hpp"
#include "iotviomanager/vio_data_type.h"
#include "iotviomanager/ring_queue.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

using hobot::vision::PymImageFrame;
using hobot::vision::SrcImageFrame;

class VpsModule {
 public:
  VpsModule() = delete;
  VpsModule(const int &pipe_id, const IotVioCfg &vio_cfg,
      const IotVinParams &params) :
    pipe_id_(pipe_id), vio_cfg_(vio_cfg), vin_params_(params) {}
  VpsModule(const int &pipe_id, const IotVioCfg &vio_cfg) :
    pipe_id_(pipe_id), vio_cfg_(vio_cfg) {}
  ~VpsModule() {}

  int FbInit();
  int FbDeInit();
  int VpsInit();
  int VpsDeInit();
  int VpsStart();
  int VpsStop();
  int VpsGetVinParams();
  int VpsGetInfo(int info_type, void *buf);
  int VpsSetInfo(int info_type, void *buf);
  int VpsFreeInfo(int info_type, void *buf);
  int SetGdcInfo();
  int GetGdcInfo(void *buf);
  int VpsCreatePymThread();
  int VpsDestoryPymThread();
  int VpsCreateFbThread();
  int VpsDestoryFbThread();
  void VpsConvertPymInfo(void *pym_buf, PymImageFrame &pym_img);
  void VpsConvertSrcInfo(void *src_buf, SrcImageFrame &src_img);
  void *VpsCreatePymAddrInfo();
  void *VpsCreateSrcAddrInfo();

 private:
  int HbGetGdcData(const char *gdc_name, char **gdc_data, int *gdc_len);
  int HbSetGdcInfo(int gdc_idx, int pipe_id, int vin_vps_mode,
      const IotVioGdcInfo &info);
  int HbDumpToFile2Plane(char *filename, char *src_buf,
      char *src_buf1, unsigned int size, unsigned int size1,
      int width, int height, int stride);
  int HbDumpPymData(int grp_id, int pym_chn, pym_buffer_t *out_pym_buf);
  int HbDumpPymLayerData(int grp_id, int pym_chn, int layer,
      pym_buffer_t *out_pym_buf);
  int HbDumpIpuData(int grp_id, int ipu_chn, hb_vio_buffer_t *out_ipu_buf);
  int HbCheckPymData(IotPymInfo &pym_info);
  int HbChnInfoInit();
  void HbGetPymDataThread();
  void HbGetFbDataThread();
  int HbVpsCreateGrp(int pipe_id, int grp_w, int grp_h, int grp_depth);
  int HbPymTimeoutWait(uint64_t timeout_ms);
  int HbGetVpsFrameDepth();
  int HbManagerPymBuffer(int max_buf_num);
  void HbAllocPymBuffer();
  void HbFreePymBuffer();


 private:
  int pipe_id_;
  IotVioCfg vio_cfg_ = { 0 };
  IotVinParams vin_params_ = { 0 };
  IotVioChnInfo vio_chn_info_ = { 0 };
  std::queue<hb_vio_buffer_t*> fb_queue_;
  hb_vio_buffer_t *feedback_buf_;
  std::shared_ptr<std::thread> fb_thread_ = nullptr;
  std::shared_ptr<std::thread> pym_thread_ = nullptr;
  std::shared_ptr<VioRingQueue<IotPymInfo>> pym_rq_ = nullptr;
  int fb_start_flag_ = false;
  int pym_start_flag_ = false;
  int gdc_pipe_id_[MAX_GDC_NUM] = {-1, -1};
  sem_t pym_sem_;
  int consumed_pym_buffers_ = 0;
  std::mutex mutex_;
  IotPymInfo vio_pym_info_ = { 0 };
  int dump_index_ = 0;
  int vps_dump_num_ = 0;
  int vb_cache_en_ = 0;
};

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_VPS_MODULE_H_
