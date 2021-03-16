/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-08-12
 * @Version: v1.0.0
 * @Brief: IOT VPS Module for Horizon VIO System.
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <chrono>
#ifdef __cplusplus
extern "C" {
#include "hb_sys.h"
#include "hb_mipi_api.h"
#include "hb_vin_api.h"
#include "hb_vio_interface.h"
#include "hb_vps_api.h"
#include "hb_vp_api.h"
}
#endif
#include "iotviomanager/vio_data_type.h"
#include "iotviomanager/ring_queue.h"
#include "iotviomanager/vpsmodule.h"
#include "iotviomanager/viopipemanager.h"
#include "iotviomanager/violog.h"
#include "hobotlog/hobotlog.hpp"

const char *kIotTypeInfo[] = {
  NULL,  // 0
  TYPE_NAME(IOT_VIO_SRC_INFO),  // 1
  TYPE_NAME(IOT_VIO_PYM_INFO),  // 2
  TYPE_NAME(IOT_VIO_SRC_MULT_INFO),  // 3
  TYPE_NAME(IOT_VIO_PYM_MULT_INFO),  // 4
  TYPE_NAME(IOT_VIO_FEEDBACK_SRC_INFO),  // 5
  TYPE_NAME(IOT_VIO_FEEDBACK_FLUSH),  // 6
  TYPE_NAME(IOT_VIO_FEEDBACK_PYM_PROCESS),  // 7
  TYPE_NAME(IOT_VIO_CHN_INFO),  // 8
};

#define CHECK_TYPE_VALID(type)\
  do {\
    uint32_t iot_type_num =\
    static_cast<uint32_t>(sizeof(kIotTypeInfo) / sizeof(char*));\
    if (type < 0 || type > iot_type_num) {\
      pr_err("check type valid falied, type:%d\n", type);\
      return -1;\
    }\
  } while (0)

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

int VpsModule::FbInit() {
  int ret = -1;
  hb_vio_buffer_t *fb_buf = nullptr;
  int fb_buf_num = vio_cfg_.vps_cfg.fb_info.buf_num;
  uint32_t fb_width   = vio_cfg_.vps_cfg.fb_info.width;
  uint32_t fb_height  = vio_cfg_.vps_cfg.fb_info.height;
  uint32_t size_y     = fb_width * fb_height;
  uint32_t size_uv    = size_y / 2;

  LOGI << "Enter FbInit," << " fb_buf_num: " << fb_buf_num
    << " fb_width: " << fb_width << " fb_height: " << fb_height;
  feedback_buf_ = static_cast<hb_vio_buffer_t*>\
                           (malloc(sizeof(hb_vio_buffer_t) * fb_buf_num));
  if (feedback_buf_ == NULL) {
    LOGE << "feedback buffer malloc failed";
    return -1;
  }
  fb_buf = feedback_buf_;

  VioPipeManager &manager = VioPipeManager::Get();
  ret = manager.VbInit();
  if (ret) {
    LOGE << "vpsmodule feedback vb init failed";
    return ret;
  }
  vb_cache_en_ = 1;
  for (int i = 0; i < fb_buf_num; i++) {
    memset(&fb_buf[i], 0, sizeof(hb_vio_buffer_t));
    ret = manager.AllocVbBuf2Lane(i, &fb_buf[i], size_y,
        size_uv, vb_cache_en_);
    if (ret) {
      LOGE << "alloc vb buffer 2 land failed, ret: " << ret;
      return ret;
    }
    fb_buf[i].img_info.planeCount = 2;
    fb_buf[i].img_info.img_format = 8;
    fb_buf[i].img_addr.width = fb_width;
    fb_buf[i].img_addr.height = fb_height;
    fb_buf[i].img_addr.stride_size = fb_width;
    fb_queue_.push(&fb_buf[i]);
  }

  LOGI << "fb_queue size: " << fb_queue_.size()
    << " pipe_id: " << pipe_id_;
  return 0;
}

int VpsModule::FbDeInit() {
  int ret = -1;
  hb_vio_buffer_t *buffer = nullptr;
  int cnt = fb_queue_.size();

  LOGI << "Enter FbDeInit";
  VioPipeManager &manager = VioPipeManager::Get();
  for (int i = 0; i < cnt; i++) {
    buffer = fb_queue_.front();
    fb_queue_.pop();
    if (buffer == NULL) {
      LOGE << "feedback queue buf index: " << i << " is nullptr";
      return -1;
    }
    ret = manager.FreeVbBuf2Lane(i, buffer);
    if (ret) {
      LOGE << "free vb buffer 2 lane failed";
    }
  }

  if (feedback_buf_) {
    free(feedback_buf_);
  }
  ret = manager.VbDeInit();
  if (ret) {
    LOGE <<  "vp exit error!";
    return ret;
  }

  return 0;
}

int VpsModule::HbGetVpsFrameDepth() {
  int frame_depth = 0;
  IotVioGdcInfo gdc_info = { 0 };
  IotGdcType_E gdc_src_type = kGDC_TYPE_INVALID;

  for (int gdc_idx = 0; gdc_idx < MAX_GDC_NUM; gdc_idx++) {
    gdc_info = vio_cfg_.vps_cfg.gdc_info[gdc_idx];
    gdc_src_type = static_cast<IotGdcType_E>(gdc_info.src_type);
    if (gdc_info.gdc_en == 1) {
      if (gdc_src_type == kISP_DDR_GDC) {
        frame_depth = gdc_info.frame_depth;
        break;
      }
    }
  }

  return frame_depth;
}

int VpsModule::HbVpsCreateGrp(int pipe_id, int grp_w, int grp_h,
    int grp_depth) {
  int ret = -1;
  VPS_GRP_ATTR_S grp_attr;

  memset(&grp_attr, 0, sizeof(VPS_GRP_ATTR_S));
  grp_attr.maxW = grp_w;
  grp_attr.maxH = grp_h;
  grp_attr.frameDepth = grp_depth;
  LOGD << "vps create group, grp_w: " << grp_w
    << " grp_h: " << grp_h << " grp_depth: " << grp_depth;
  ret = HB_VPS_CreateGrp(pipe_id, &grp_attr);
  if (ret) {
    LOGE << "HB_VPS_CreateGrp error!!!";
    return ret;
  } else {
    LOGI << "created a group ok:GrpId: " << pipe_id;
  }

  return 0;
}

int VpsModule::SetGdcInfo() {
  int ret = -1;
  int vin_vps_mode = vio_cfg_.vin_vps_mode;
  int gdc_pipe_id = pipe_id_;
  IotVioGdcInfo gdc_info = { 0 };

  /* set gdc */
  for (int gdc_idx = 0; gdc_idx < MAX_GDC_NUM; gdc_idx++) {
    gdc_info = vio_cfg_.vps_cfg.gdc_info[gdc_idx];
    if (gdc_info.gdc_en == 1) {
      ret = HbSetGdcInfo(gdc_idx, gdc_pipe_id, vin_vps_mode, gdc_info);
      if (ret) {
        LOGE << "Set GDC failed!, index: " << gdc_idx << "ret: " << ret;
        return ret;
      }
    }
  }

  return 0;
}

int VpsModule::GetGdcInfo(void *buf) {
  int ret = -1;
  IotVioGdcInfo gdc_info = { 0 };
  int gdc_pipe_id = -1;
  int gdc_chn = -1;
  int timeout = 2000;
  IotGdcType_E gdc_src_type = kGDC_TYPE_INVALID;

  hb_vio_buffer_t *gdc_buf = reinterpret_cast<hb_vio_buffer_t*>(buf);
  /* get gdc info */
  for (int gdc_idx = 0; gdc_idx < MAX_GDC_NUM; gdc_idx++) {
    gdc_info = vio_cfg_.vps_cfg.gdc_info[gdc_idx];
    gdc_src_type = static_cast<IotGdcType_E>(gdc_info.src_type);
    if (gdc_info.gdc_en == 1 && gdc_src_type == kPYM_DDR_GDC) {
      gdc_pipe_id = GDC_PIPE_ID + gdc_idx;
      gdc_chn = 0;
      ret = HB_VPS_GetChnFrame(gdc_pipe_id, gdc_chn, gdc_buf, timeout);
      if (ret) {
        LOGE << "Get Chn GDC frame failed!, index: "
          << gdc_idx << "ret: " << ret;
        return ret;
      }
    }
  }

  return 0;
}

int VpsModule::VpsInit() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int cam_en = vio_cfg_.cam_en;
  int grp_width, grp_height, grp_frame_depth;
  int chn_num, ipu_chn_en, pym_chn_en, ipu_frame_depth;
  int scale_en, scale_w, scale_h;
  int roi_en, roi_x, roi_y, roi_w, roi_h;
  int ds2_roi_en, ds2_roi_x, ds2_roi_y, ds2_roi_w, ds2_roi_h;
  int ipu_ds2_en = 0;
  int ipu_ds2_crop_en = 0;
  VPS_CHN_ATTR_S chn_attr;
  VPS_PYM_CHN_ATTR_S pym_chn_attr;

  LOGI << "Enter vpsmodule init," << " pipe_id: " << pipe_id_
    << " cam_en: " << cam_en;
  HbChnInfoInit();
  vps_dump_num_ = vio_cfg_.vps_cfg.vps_dump_num;
  if (cam_en == 1) {
    grp_width = vin_params_.pipe_info.stSize.width;
    grp_height = vin_params_.pipe_info.stSize.height;
  } else {
    grp_width = vio_cfg_.vps_cfg.fb_info.width;
    grp_height = vio_cfg_.vps_cfg.fb_info.height;
  }
  HOBOT_CHECK(grp_width > 0) << "group width is valid!";
  HOBOT_CHECK(grp_height > 0) << "group height is valid!";
  grp_frame_depth = HbGetVpsFrameDepth();
  ret = HbVpsCreateGrp(pipe_id, grp_width, grp_height, grp_frame_depth);
  HOBOT_CHECK(ret == 0) << "HbVpsCreateGrp failed!";
  ret = SetGdcInfo();
  HOBOT_CHECK(ret == 0) << "HbVpsCreateGrp failed!";

  chn_num = vio_cfg_.vps_chn_num;
  for (int chn_idx = 0; chn_idx < chn_num; chn_idx++) {
    /* 1. set ipu chn */
    ipu_chn_en = vio_cfg_.chn_cfg[chn_idx].ipu_chn_en;
    pym_chn_en = vio_cfg_.chn_cfg[chn_idx].pym_chn_en;
    ipu_frame_depth = vio_cfg_.chn_cfg[chn_idx].ipu_frame_depth;
    roi_en = vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_en;
    roi_x = vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_x;
    roi_y = vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_y;
    roi_w = vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_w;
    roi_h = vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_h;
    scale_w = vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_w;
    scale_h = vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_h;
    if (chn_idx == 2) {
      ds2_roi_en = roi_en;
      ds2_roi_x = roi_x;
      ds2_roi_y = roi_y;
      ds2_roi_w = roi_w;
      ds2_roi_y = roi_h;
    }

    /* set ipu chn */
    if (ipu_chn_en == 1) {
      memset(&chn_attr, 0, sizeof(VPS_CHN_ATTR_S));
      vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_en = 1;
      scale_en = 1;
      chn_attr.enScale = scale_en;
      chn_attr.width = scale_w;
      chn_attr.height = scale_h;
      chn_attr.frameDepth = ipu_frame_depth;
      ret = HB_VPS_SetChnAttr(pipe_id, chn_idx, &chn_attr);
      if (ret) {
        LOGE << "HB_VPS_SetChnAttr error!!!";
        return ret;
      } else {
        LOGI << "vps set ipu chn Attr ok: GrpId: " << pipe_id
          << " chn_id: " << chn_idx;
      }
      if (roi_en == 1) {
        int crop_chn = chn_idx;
        if (chn_idx == 6 && ipu_ds2_en == 0) {
          /* set ipu chn2 attr */
          crop_chn = 2;
          memset(&chn_attr, 0, sizeof(VPS_CHN_ATTR_S));
          chn_attr.enScale = scale_en;
          chn_attr.width = scale_w;
          chn_attr.height = scale_h;
          chn_attr.frameDepth = 0;
          ret = HB_VPS_SetChnAttr(pipe_id, crop_chn, &chn_attr);
          if (ret) {
            LOGE << "HB_VPS_SetChnAttr error!!!";
            return ret;
          } else {
            LOGI << "vps set ipu chn Attr ok: GrpId: " << pipe_id
              << " chn_id: " << crop_chn;
          }
        }
        if (chn_idx == 6 && ipu_ds2_crop_en == 1) {
          LOGW << "ipu chn6 is not need set crop, because chn2 has been"
            << " crop(chn6 and chn2 is shared crop chn settings)";
          HOBOT_CHECK(ds2_roi_en == roi_en);
          HOBOT_CHECK(ds2_roi_x == roi_x);
          HOBOT_CHECK(ds2_roi_y == roi_y);
          HOBOT_CHECK(ds2_roi_w == roi_w);
          HOBOT_CHECK(ds2_roi_h == roi_h);
        } else {
          VPS_CROP_INFO_S crop_info;
          memset(&crop_info, 0, sizeof(VPS_CROP_INFO_S));
          crop_info.en = roi_en;
          crop_info.cropRect.width = roi_w;
          crop_info.cropRect.height = roi_h;
          crop_info.cropRect.x = roi_x;
          crop_info.cropRect.y = roi_y;
          LOGI << "crop en: " << crop_info.en
            << " crop width: " << crop_info.cropRect.width
            << " crop height: " << crop_info.cropRect.height
            << " crop x: " << crop_info.cropRect.x
            << " crop y: " << crop_info.cropRect.y;
          ret = HB_VPS_SetChnCrop(pipe_id, crop_chn, &crop_info);
          if (ret) {
            LOGE << "HB_VPS_SetChnCrop error!!!";
            return ret;
          } else {
            LOGI << "vps set ipu crop ok: GrpId: " << pipe_id
              << " chn_id: " << chn_idx;
          }
          if (chn_idx == 2) {
            ipu_ds2_crop_en = 1;
          }
        }
      }
      if (chn_idx == 2) {
        ipu_ds2_en = 1;
      }
      HB_VPS_EnableChn(pipe_id, chn_idx);
    }
    /* set pym chn */
    if (pym_chn_en == 1) {
      memset(&pym_chn_attr, 0, sizeof(VPS_PYM_CHN_ATTR_S));
      memcpy(&pym_chn_attr, &vio_cfg_.pym_cfg,
          sizeof(VPS_PYM_CHN_ATTR_S));
      ret = HB_VPS_SetPymChnAttr(pipe_id, chn_idx, &pym_chn_attr);
      if (ret) {
        LOGE << "HB_VPS_SetPymChnAttr error!!!";
        return ret;
      } else {
        LOGI << "vps set pym chn Attr ok: GrpId: " << pipe_id
          << " chn_id: " << chn_idx;
      }
      HB_VPS_EnableChn(pipe_id, chn_idx);
      LOGD << "ds5 factor: " << pym_chn_attr.ds_info[5].factor
        << " roi_x: " << pym_chn_attr.ds_info[5].roi_x
        << " roi_y: " << pym_chn_attr.ds_info[5].roi_y
        << " roi_width: " << pym_chn_attr.ds_info[5].roi_width
        << " roi_height: " << pym_chn_attr.ds_info[5].roi_height;

      LOGD << "ds6 factor: " << pym_chn_attr.ds_info[6].factor
        << " roi_x: " << pym_chn_attr.ds_info[6].roi_x
        << " roi_y: " << pym_chn_attr.ds_info[6].roi_y
        << " roi_width: " << pym_chn_attr.ds_info[6].roi_width
        << " roi_height: " << pym_chn_attr.ds_info[6].roi_height;
    }
  }

  sem_init(&pym_sem_, 0, 0);
  return 0;
}

int VpsModule::VpsDeInit() {
  int ret = -1;
  int gdc_pipe_id, gdc_id;
  int pipe_id = pipe_id_;

  ret = HB_VPS_DestroyGrp(pipe_id);
  if (ret < 0) {
    LOGE << "vps deinit failed!";
    return ret;
  }

  /* gdc vpsgroup deinit*/
  for (int gdc_idx = 0; gdc_idx < MAX_GDC_NUM; gdc_idx++) {
    gdc_pipe_id = gdc_pipe_id_[gdc_idx];
    gdc_id = GDC_PIPE_ID + gdc_idx;
    if (gdc_pipe_id == gdc_id) {
      ret = HB_VPS_DestroyGrp(pipe_id);
      if (ret < 0) {
        LOGE << "vps deinit failed!, gdc_pipe_id: " << gdc_pipe_id;
        return ret;
      }
    }
  }
  sem_destroy(&pym_sem_);
  return 0;
}

int VpsModule::VpsCreateFbThread() {
  LOGD << "Enter VpsCreateFbThread, pipe_id: " << pipe_id_
    << " fb_start_flag: " << fb_start_flag_;

  if (fb_start_flag_ == true) {
    LOGE << "Vps fb thread has been create";
    return -1;
  }

  if (fb_thread_ == nullptr) {
    fb_thread_ = std::make_shared<std::thread>(
        &VpsModule::HbGetFbDataThread, this);
    fb_start_flag_ = true;
  }

  return 0;
}

int VpsModule::VpsDestoryFbThread() {
  LOGI << "Enter VpsDestoryFbThread, pipe_id: " << pipe_id_
    << " fb_start_flag: " << fb_start_flag_;

  if (fb_start_flag_ == false) {
    LOGE << "Vps fb thread has been destory";
    return -1;
  }

  if (fb_thread_ != nullptr) {
    fb_start_flag_ = false;
    fb_thread_->join();
  } else {
    LOGE << "feedback thread is nullptr!!!";
    return -1;
  }

  LOGI << "Quit VpsDestoryFbThread, pipe_id: " << pipe_id_
    << " start_flag: " << fb_start_flag_;
  return 0;
}

int VpsModule::VpsCreatePymThread() {
  LOGD << "Enter VpsCreatePymThread, pipe_id: " << pipe_id_
    << " start_flag: " << pym_start_flag_;

  if (pym_start_flag_ == true) {
    LOGE << "Vps module has been create";
    return -1;
  }

  if (pym_thread_ == nullptr) {
    pym_thread_ = std::make_shared<std::thread>(
        &VpsModule::HbGetPymDataThread, this);
    pym_start_flag_ = true;
  }

  return 0;
}

int VpsModule::VpsDestoryPymThread() {
  LOGI << "Enter VpsDestoryPymThread, pipe_id: " << pipe_id_
    << " start_flag: " << pym_start_flag_;

  if (pym_start_flag_ == false) {
    LOGE << "Vps module has been destory";
    return -1;
  }

  if (pym_thread_ != nullptr) {
    pym_start_flag_ = false;
    pym_thread_->join();
    pym_thread_ = nullptr;
  } else {
    LOGE << "pym thread is nullptr!!!";
    return -1;
  }

  LOGI << "Quit VpsDestoryPymThread, pipe_id: " << pipe_id_
    << " start_flag: " << pym_start_flag_;
  return 0;
}

int VpsModule::VpsStart() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int inner_buf_en = vio_cfg_.vps_cfg.fb_info.inner_buf_en;

  LOGD << "Enter Vps Start, pipe_id: " << pipe_id;
  ret = HB_VPS_StartGrp(pipe_id);
  if (ret) {
    LOGE << "pipe_id: " <<  pipe_id
      << " HB_VPS_StartGrp error, ret: " << ret;
    return ret;
  }
  if (inner_buf_en == 1) {
    ret = VpsCreateFbThread();
    if (ret) {
      LOGE << "Vps create pym thread failed!!!";
      return ret;
    }
  }
  ret = VpsCreatePymThread();
  if (ret) {
    LOGE << "Vps create pym thread failed!!!";
    return ret;
  }

  return 0;
}

int VpsModule::VpsStop() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int inner_buf_en = vio_cfg_.vps_cfg.fb_info.inner_buf_en;

  LOGI << "Enter vps module stop, pipe_id: " << pipe_id;
  ret = VpsDestoryPymThread();
  if (ret) {
    LOGE << "Vps destory pym thread failed!!!";
    return ret;
  }
  if (inner_buf_en == 1) {
    ret = VpsDestoryFbThread();
    if (ret) {
      LOGE << "Vps destory feedback thread failed!!!";
      return ret;
    }
  }
  LOGI << "HB_VPS_StopGrp, pipe_id: " << pipe_id;
  ret = HB_VPS_StopGrp(pipe_id);
  if (ret) {
    LOGE << "HB_VPS_StopGrp error!!!";
    return ret;
  } else {
    LOGI << "stop grp ok: grp_id = " << pipe_id;
  }

  LOGI << "Quit vps module stop, pipe_id: " << pipe_id;
  return 0;
}

int VpsModule::HbGetGdcData(const char *gdc_name,
    char **gdc_data, int *gdc_len) {
  LOGD << "start to set GDC";
  std::ifstream ifs(gdc_name);
  if (!ifs.is_open()) {
    LOGE << "GDC file open failed!";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  auto len = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  auto buf = new char[len];
  ifs.read(buf, len);

  *gdc_data = buf;
  *gdc_len = len;

  return 0;
}

int VpsModule::HbSetGdcInfo(int gdc_idx, int pipe_id, int vin_vps_mode,
    const IotVioGdcInfo &info) {
  int ret = 0;
  char *gdc_data = nullptr;
  int gdc_pipe_id = pipe_id;
  int gdc_len = 0;
  int gdc_w = info.gdc_w;
  int gdc_h = info.gdc_h;
  IotGdcType_E gdc_src_type = static_cast<IotGdcType_E>(info.src_type);
  int gdc_frame_depth = info.frame_depth;
  int gdc_chn = info.bind_ipu_chn;
  const char *gdc_path = info.file_path.c_str();
  ROTATION_E gdc_rotate = static_cast<ROTATION_E>(info.rotate);

  LOGD << "gdc_idx: " << gdc_idx << " vin_vps_mode: " << vin_vps_mode
    << " gdc_src_type: " << gdc_src_type << " gdc_rotate: " << gdc_rotate
    << " gdc_path: " << gdc_path;

  if (gdc_src_type == kISP_DDR_GDC) {
    if (vin_vps_mode == VIN_ONLINE_VPS_ONLINE ||
        vin_vps_mode == VIN_OFFLINE_VPS_ONLINE ||
        vin_vps_mode == VIN_SIF_ONLINE_DDR_ISP_ONLINE_VPS_ONLINE ||
        vin_vps_mode == VIN_SIF_OFFLINE_ISP_OFFLINE_VPS_ONLINE ||
        vin_vps_mode == VIN_FEEDBACK_ISP_ONLINE_VPS_ONLINE ||
        vin_vps_mode == VIN_SIF_VPS_ONLINE) {
      LOGE << "vin_vps_mode error!!!"
        << "vps need be set offline mode if gdc enable";
      return -1;
    }
  }

  ret = HbGetGdcData(gdc_path, &gdc_data, &gdc_len);
  if (ret) {
    LOGE << "Hb get gdc data failed!";
    return -1;
  }
  LOGD << "gdc_data addr: " << reinterpret_cast<void *>(gdc_data)
    << " gdc_len: " << gdc_len;

  switch (gdc_src_type) {
    case kISP_DDR_GDC: {
      ret = HB_VPS_SetGrpGdc(gdc_pipe_id, gdc_data, gdc_len, gdc_rotate);
      if (ret) {
        LOGE << "HB_VPS_SetGrpGdc error!!!";
      } else {
        LOGI << "HB_VPS_SetGrpGdc ok: gdc_pipe_id = " << gdc_pipe_id;
      }
      break;
    }
    case kIPU_CHN_DDR_GDC: {
      ret =  HB_VPS_SetChnGdc(gdc_pipe_id, gdc_chn, gdc_data, gdc_len,
          gdc_rotate);
      if (ret) {
        LOGE << "HB_VPS_SetChnGdc error!!!";
      } else {
        LOGI << "HB_VPS_SetChnGdc ok, gdc_pipe_id: " << gdc_pipe_id;
      }
      break;
    }
    case kPYM_DDR_GDC: {
      gdc_pipe_id_[gdc_idx] = GDC_PIPE_ID + gdc_idx;
      gdc_pipe_id = gdc_pipe_id_[gdc_idx];
      ret = HbVpsCreateGrp(gdc_pipe_id, gdc_w, gdc_h, gdc_frame_depth);
      if (ret) {
        LOGE << "HbVpsCreateGrp failed!!!";
      }

      ret = HB_VPS_SetGrpGdc(pipe_id, gdc_data, gdc_len, gdc_rotate);
      if (ret) {
        LOGE << "HB_VPS_SetGrpGdc error!!!";
      } else {
        LOGI << "HB_VPS_SetGrpGdc ok: gdc pipe_id = " << gdc_pipe_id;
      }
      break;
    }
    default:
      LOGE << "not support gdc type:" << gdc_src_type;
      break;
  }
  free(gdc_data);

  return ret;
}

int VpsModule::HbDumpToFile2Plane(char *filename, char *src_buf,
    char *src_buf1, unsigned int size, unsigned int size1,
    int width, int height, int stride) {
  FILE *yuvFd = NULL;
  char *buffer = NULL;
  int i = 0;

  yuvFd = fopen(filename, "w+");

  if (yuvFd == NULL) {
    LOGE << "open file: " << filename << "failed";
    return -1;
  }

  buffer = reinterpret_cast<char *>(malloc(size + size1));

  if (buffer == NULL) {
    LOGE << "malloc falied";
    fclose(yuvFd);
    return -1;
  }

  if (width == stride) {
    memcpy(buffer, src_buf, size);
    memcpy(buffer + size, src_buf1, size1);
  } else {
    // jump over stride - width Y
    for (i = 0; i < height; i++) {
      memcpy(buffer+i*width, src_buf+i*stride, width);
    }

    // jump over stride - width UV
    for (i = 0; i < height/2; i++) {
      memcpy(buffer+size+i*width, src_buf1+i*stride, width);
    }
  }

  fwrite(buffer, 1, size + size1, yuvFd);

  fflush(yuvFd);

  if (yuvFd)
    fclose(yuvFd);
  if (buffer)
    free(buffer);

  pr_debug("filedump(%s, size(%d) is successed!!\n", filename, size);

  return 0;
}

int VpsModule::HbDumpPymData(int grp_id, int pym_chn,
    pym_buffer_t *out_pym_buf) {
  int i;
  char file_name[100] = {0};

  for (i = 0; i < 6; i++) {
    snprintf(file_name,
        sizeof(file_name),
        "grp%d_chn%d_pym_out_basic_layer_DS%d_%d_%d.yuv",
        grp_id, pym_chn, i * 4,
        out_pym_buf->pym[i].width,
        out_pym_buf->pym[i].height);
    HbDumpToFile2Plane(
        file_name,
        out_pym_buf->pym[i].addr[0],
        out_pym_buf->pym[i].addr[1],
        out_pym_buf->pym[i].width * out_pym_buf->pym[i].height,
        out_pym_buf->pym[i].width * out_pym_buf->pym[i].height / 2,
        out_pym_buf->pym[i].width,
        out_pym_buf->pym[i].height,
        out_pym_buf->pym[i].stride_size);
    for (int j = 0; j < 3; j++) {
      snprintf(file_name,
          sizeof(file_name),
          "grp%d_chn%d_pym_out_roi_layer_DS%d_%d_%d.yuv",
          grp_id, pym_chn, i * 4 + j + 1,
          out_pym_buf->pym_roi[i][j].width,
          out_pym_buf->pym_roi[i][j].height);
      if (out_pym_buf->pym_roi[i][j].width != 0)
        HbDumpToFile2Plane(
            file_name,
            out_pym_buf->pym_roi[i][j].addr[0],
            out_pym_buf->pym_roi[i][j].addr[1],
            out_pym_buf->pym_roi[i][j].width *
            out_pym_buf->pym_roi[i][j].height,
            out_pym_buf->pym_roi[i][j].width *
            out_pym_buf->pym_roi[i][j].height / 2,
            out_pym_buf->pym_roi[i][j].width,
            out_pym_buf->pym_roi[i][j].height,
            out_pym_buf->pym_roi[i][j].stride_size);
    }
    snprintf(file_name,
        sizeof(file_name),
        "grp%d_chn%d_pym_out_us_layer_US%d_%d_%d.yuv",
        grp_id, pym_chn, i,
        out_pym_buf->us[i].width,
        out_pym_buf->us[i].height);
    if (out_pym_buf->us[i].width != 0)
      HbDumpToFile2Plane(
          file_name,
          out_pym_buf->us[i].addr[0],
          out_pym_buf->us[i].addr[1],
          out_pym_buf->us[i].width * out_pym_buf->us[i].height,
          out_pym_buf->us[i].width * out_pym_buf->us[i].height / 2,
          out_pym_buf->us[i].width,
          out_pym_buf->us[i].height,
          out_pym_buf->us[i].stride_size);
  }

  return 0;
}

int VpsModule::HbDumpPymLayerData(int grp_id, int pym_chn,
    int layer, pym_buffer_t *out_pym_buf) {
  char file_name[100] = {0};
  address_info_t pym_addr;
  int y_len, uv_len;

  uint32_t frame_id = out_pym_buf->pym_img_info.frame_id;
  uint64_t ts = out_pym_buf->pym_img_info.time_stamp;

  pr_debug("ts:%lu, frameID:%d\n", ts, frame_id);

  if (out_pym_buf == NULL) {
    LOGE << "out_pym_buf is nullptr";
    return -1;
  }

  if (layer % 4 == 0) {
    pym_addr = out_pym_buf->pym[layer / 4];
    snprintf(file_name,
        sizeof(file_name),
        "%d_grp%d_chn%d_pym_out_basic_layer_DS%d_%d_%d.yuv",
        dump_index_++, grp_id, pym_chn, layer,
        pym_addr.width, pym_addr.height);
  } else {
    pym_addr = out_pym_buf->pym_roi[layer / 4][layer % 4 - 1];
    snprintf(file_name,
        sizeof(file_name),
        "%d_grp%d_chn%d_pym_out_roi_layer_DS%d_%d_%d.yuv",
        dump_index_++, grp_id, pym_chn, layer,
        pym_addr.width, pym_addr.height);
  }

  if (pym_addr.width == 0 ||
      pym_addr.height == 0) {
    LOGE << "pym_width: " << pym_addr.width
      << " pym_height" << pym_addr.height << "error!!!";
    return -1;
  }

  y_len = pym_addr.width * pym_addr.height;
  uv_len = y_len / 2;
  HbDumpToFile2Plane(file_name,
      pym_addr.addr[0],
      pym_addr.addr[1],
      y_len, uv_len,
      pym_addr.width,
      pym_addr.height,
      pym_addr.stride_size);

  return 0;
}

int VpsModule::HbDumpIpuData(int grp_id, int ipu_chn,
    hb_vio_buffer_t *out_ipu_buf) {
  char file_name[100] = {0};

  /* HbPrintVioBufInfo(out_ipu_buf); */
  snprintf(file_name,
      sizeof(file_name),
      "grp%d_chn%d_%d_%d.yuv",
      grp_id, ipu_chn,
      out_ipu_buf->img_addr.width,
      out_ipu_buf->img_addr.height);

  HbDumpToFile2Plane(
      file_name,
      out_ipu_buf->img_addr.addr[0],
      out_ipu_buf->img_addr.addr[1],
      out_ipu_buf->img_addr.width * out_ipu_buf->img_addr.height,
      out_ipu_buf->img_addr.width * out_ipu_buf->img_addr.height / 2,
      out_ipu_buf->img_addr.width,
      out_ipu_buf->img_addr.height,
      out_ipu_buf->img_addr.stride_size);

  return 0;
}

int VpsModule::HbCheckPymData(IotPymInfo &pym_info) {
  int i;
  int ret = 0;
  address_info_t pym_addr;
  pym_buffer_t *pym_buf = &pym_info.pym_buf;
  uint64_t ts_cur;
  int frame_id;

  frame_id = pym_info.pym_buf.pym_img_info.frame_id;
  ts_cur = pym_info.pym_buf.pym_img_info.time_stamp;
  pr_debug("pipe_id:%d buf_index[%d] pym_chn:%d"
      " frame_id:%d ts_cur: %lu \n",
      pym_info.grp_id,
      pym_info.buf_index,
      pym_info.pym_chn,
      frame_id, ts_cur);

  for (i = 0; i < DOWN_SCALE_MAX; ++i) {
    if (i % 4 == 0) {  // base pym layer
      pym_addr = pym_buf->pym[i / 4];
      if (pym_addr.width == 0 || pym_addr.height == 0) {
        ret = -1;
      }
    } else {  // roi pym layer
      pym_addr = pym_buf->pym_roi[i / 4][i % 4 - 1];
    }
    pr_debug("layer:%d, width:%d, height:%d, stride_size:%d, "
        "y_addr:%p, uv_addr:%p\n",
        i,
        pym_addr.width, pym_addr.height, pym_addr.stride_size,
        pym_addr.addr[0], pym_addr.addr[1]);
    if (ret) {
      LOGE << "vio check pym failed"
        << " pym_layer: " << i
        << " width: " << pym_addr.width
        << " height: " << pym_addr.height;
      return ret;
    }
  }

  return 0;
}

int VpsModule::HbChnInfoInit() {
  int chn_idx, ipu_chn_index, pym_chn_index;
  int ipu_chn_en, pym_chn_en;

  /* 1. init ipu and pym chn value */
  vio_chn_info_.ipu_chn_num = 0;
  vio_chn_info_.pym_chn_num = 0;
  for (chn_idx = 0; chn_idx < MAX_CHN_NUM; chn_idx++) {
    vio_chn_info_.ipu_valid_chn[chn_idx] = -1;
    vio_chn_info_.pym_valid_chn[chn_idx] = -1;
  }

  /* 2. get valid ipu and pym chn, mult pipe is in one object */
  ipu_chn_index = 0;
  pym_chn_index = 0;
  for (chn_idx = 0; chn_idx < MAX_CHN_NUM; chn_idx++) {
    ipu_chn_en = vio_cfg_.chn_cfg[chn_idx].ipu_chn_en;
    pym_chn_en = vio_cfg_.chn_cfg[chn_idx].pym_chn_en;

    if (pym_chn_en == 1) {
      vio_chn_info_.pym_valid_chn[pym_chn_index] = chn_idx;
      LOGD << "pipe_id: " << pipe_id_
        << " pym_chn_index: " << pym_chn_index
        << " pym_chn_valid: " << chn_idx;
      pym_chn_index++;
      // if this pym chn enable, vps can not get same ipu chn data
      continue;
    }
    if (ipu_chn_en == 1) {
      vio_chn_info_.ipu_valid_chn[ipu_chn_index] = chn_idx;
      LOGD << "pipe_id: " << pipe_id_
        << " ipu_chn_index: " << ipu_chn_index
        << " ipu_chn_valid: " << chn_idx;
      ipu_chn_index++;
    }
  }
  HOBOT_CHECK(pym_chn_index <= 1) << "a group only support max a pym chn";
  vio_chn_info_.pym_chn_num = pym_chn_index;
  vio_chn_info_.ipu_chn_num = ipu_chn_index;

  return 0;
}

int VpsModule::HbPymTimeoutWait(uint64_t timeout_ms) {
  int32_t ret = -1;
  uint64_t end_time = 0;
  struct timeval currtime;
  struct timespec ts;

  if (timeout_ms != 0) {
    gettimeofday(&currtime, NULL);
    end_time = (((((uint64_t) currtime.tv_sec) * TIME_MICRO_SECOND) +
          currtime.tv_usec) + (timeout_ms * 1000));
    ts.tv_sec = (end_time / TIME_MICRO_SECOND);
    ts.tv_nsec = ((end_time % TIME_MICRO_SECOND) * 1000);
    ret = sem_timedwait(&pym_sem_, &ts);
  } else {
    ret = sem_wait(&pym_sem_);
  }

  return ret;
}

void *VpsModule::VpsCreatePymAddrInfo() {
  auto *pvio_image = reinterpret_cast<void*>(
      std::calloc(1, sizeof(pym_buffer_t)));
  if (nullptr == pvio_image) {
    LOGE << "std::calloc pym buffer failed";
    return nullptr;
  }

  return pvio_image;
}

void *VpsModule::VpsCreateSrcAddrInfo() {
  auto *pvio_image = reinterpret_cast<void*>(
      std::calloc(1, sizeof(hb_vio_buffer_t)));
  if (nullptr == pvio_image) {
    LOGE << "std::calloc pym buffer failed";
    return nullptr;
  }

  return pvio_image;
}

void VpsModule::VpsConvertPymInfo(void *pym_buf, PymImageFrame &pym_img) {
  auto pym_buffer = static_cast<pym_buffer_t*>(pym_buf);
  if (nullptr == pym_buffer) {
    return;
  }
  pym_img.ds_pym_total_layer = DOWN_SCALE_MAX;
  pym_img.us_pym_total_layer = UP_SCALE_MAX;
  pym_img.frame_id = pym_buffer->pym_img_info.frame_id;
  pym_img.time_stamp = pym_buffer->pym_img_info.time_stamp;
  pym_img.context = pym_buf;
  for (int i = 0; i < DOWN_SCALE_MAX; ++i) {
    address_info_t *pym_addr = NULL;
    if (i % 4 == 0) {
      pym_addr = reinterpret_cast<address_info_t *>(&pym_buffer->pym[i / 4]);
    } else {
      pym_addr = reinterpret_cast<address_info_t *>(
          &pym_buffer->pym_roi[i / 4][i % 4 - 1]);
    }
    pym_img.down_scale[i].width = pym_addr->width;
    pym_img.down_scale[i].height = pym_addr->height;
    pym_img.down_scale[i].stride = pym_addr->stride_size;
    pym_img.down_scale[i].y_paddr = pym_addr->paddr[0];
    pym_img.down_scale[i].c_paddr = pym_addr->paddr[1];
    pym_img.down_scale[i].y_vaddr =
        reinterpret_cast<uint64_t>(pym_addr->addr[0]);
    pym_img.down_scale[i].c_vaddr =
        reinterpret_cast<uint64_t>(pym_addr->addr[1]);
  }
  for (int i = 0; i < UP_SCALE_MAX; ++i) {
    pym_img.up_scale[i].width = pym_buffer->us[i].width;
    pym_img.up_scale[i].height = pym_buffer->us[i].height;
    pym_img.up_scale[i].stride = pym_buffer->us[i].stride_size;
    pym_img.up_scale[i].y_paddr = pym_buffer->us[i].paddr[0];
    pym_img.up_scale[i].c_paddr = pym_buffer->us[i].paddr[1];
    pym_img.up_scale[i].y_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->us[i].addr[0]);
    pym_img.up_scale[i].c_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->us[i].addr[1]);
  }
  for (int i = 0; i < DOWN_SCALE_MAIN_MAX; ++i) {
    pym_img.down_scale_main[i].width = pym_buffer->pym[i].width;
    pym_img.down_scale_main[i].height = pym_buffer->pym[i].height;
    pym_img.down_scale_main[i].stride = pym_buffer->pym[i].stride_size;
    pym_img.down_scale_main[i].y_paddr = pym_buffer->pym[i].paddr[0];
    pym_img.down_scale_main[i].c_paddr = pym_buffer->pym[i].paddr[1];
    pym_img.down_scale_main[i].y_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->pym[i].addr[0]);
    pym_img.down_scale_main[i].c_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->pym[i].addr[1]);
  }
}

void VpsModule::VpsConvertSrcInfo(void *src_buf, SrcImageFrame &src_img) {
  auto src_buffer = static_cast<hb_vio_buffer_t*>(src_buf);
  if (nullptr == src_buffer) {
    return;
  }
  src_img.src_info.width = src_buffer->img_addr.width;
  src_img.src_info.height = src_buffer->img_addr.height;
  src_img.src_info.stride = src_buffer->img_addr.stride_size;
  src_img.src_info.y_paddr = src_buffer->img_addr.paddr[0];
  src_img.src_info.c_paddr = src_buffer->img_addr.paddr[1];
  src_img.src_info.y_vaddr =
    reinterpret_cast<uint64_t>(src_buffer->img_addr.addr[0]);
  src_img.src_info.c_vaddr =
    reinterpret_cast<uint64_t>(src_buffer->img_addr.addr[1]);
}

int VpsModule::VpsGetInfo(int info_type, void *buf) {
  int ret = -1;
  int chret = -1;
  int grp_id, timeout;
  hb_vio_buffer_t *ipu_buf, *vb_buf, *fb_src_buf;
  pym_buffer_t *pym_buf;
  IotVioChnInfo *chn_info;
  int ipu_valid_chn, ipu_chn_num, pym_valid_chn, pym_chn_num;
  int vps_layer_dump;
  char *ipu_dump_env = NULL;
  int ipu_dump_en = 0;
  // pym_buffer_t out_pym_buf;

  grp_id = pipe_id_;
  vps_layer_dump = vio_cfg_.vps_cfg.vps_layer_dump;
  switch (info_type) {
    case IOT_VIO_CHN_INFO: {
      chn_info = reinterpret_cast<IotVioChnInfo*>(buf);
      *chn_info = vio_chn_info_;
      break;
    }
    case IOT_VIO_SRC_INFO: {
      ipu_buf = reinterpret_cast<hb_vio_buffer_t*>(buf);
      ipu_chn_num = vio_chn_info_.ipu_chn_num;
      if (ipu_chn_num <= 0 || ipu_chn_num >= MAX_CHN_NUM) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " ipu chn num is invalid: " << ipu_chn_num
          << " grp_id: " << grp_id;
        return -1;
      }

      for (int i = 0; i < ipu_chn_num; i++) {
        ipu_valid_chn = vio_chn_info_.ipu_valid_chn[i];
        timeout = vio_cfg_.chn_cfg[ipu_valid_chn].ipu_frame_timeout;
        ret = HB_VPS_GetChnFrame(grp_id, ipu_valid_chn, ipu_buf, timeout);
        if (ret != 0) {
          LOGE << "info_type: " << kIotTypeInfo[info_type]
            << " HB_VPS_GetChnFrame error, ret: " << ret;
          return ret;
        }
        ipu_dump_env = getenv("ipu_dump");
        ipu_dump_en = atoi(ipu_dump_env);
        if (ipu_dump_en == 1) {
          HbDumpIpuData(grp_id, ipu_valid_chn, ipu_buf);
        }
        break;
      }
      break;
    }
    case IOT_VIO_SRC_MULT_INFO: {
      break;
    }
    case IOT_VIO_PYM_INFO: {
      IotPymInfo pym_info = { 0 };
      pym_buf = reinterpret_cast<pym_buffer_t *>(buf);
      pym_chn_num = vio_chn_info_.pym_chn_num;
      pym_valid_chn = vio_chn_info_.pym_valid_chn[0];
      timeout = vio_cfg_.pym_cfg.timeout;
      if (pym_chn_num != 1) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " pym chn num is invalid: " << pym_chn_num
          << " grp_id: " << grp_id;
        return -1;
      }

      ret = HbPymTimeoutWait(timeout);
      if (ret < 0) {
        LOGE << "Wait Pym Data Timeout!!!" << " pipe_id: " << pipe_id_;
        return ret;
      }
      auto rv = pym_rq_->Pop(pym_info);
      if (!rv) {
        LOGE << "No Pym info in RingQueue!" << " pipe_id: " << pipe_id_;
        return -1;
      }
      *pym_buf = pym_info.pym_buf;
      HbAllocPymBuffer();
      chret = HbCheckPymData(pym_info);
      if (chret != 0) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " vio check pym error chret: " << chret;
        chret = HB_VPS_ReleaseChnFrame(pym_info.grp_id, pym_info.pym_chn,
            &pym_info.pym_buf);
        if (chret != 0) {
          LOGE << "info_type: " << kIotTypeInfo[info_type]
            << " HB_VPS_ReleaseChnFrame error, ret: " << ret;
          return chret;
        }
        HbFreePymBuffer();
        return chret;
      }
      if (vps_dump_num_-- > 0) {
        if (vps_layer_dump == MAX_PYM_DS_NUM) {
          // dump all pym base layer
          HbDumpPymData(grp_id, pym_valid_chn, pym_buf);
        }
        if (vps_layer_dump >=0 && vps_layer_dump < MAX_PYM_DS_NUM) {
          // dump any one pym layer(including base and roi layer)
          HbDumpPymLayerData(grp_id, pym_valid_chn, vps_layer_dump, pym_buf);
        }
      }
      break;
    }
    case IOT_VIO_PYM_MULT_INFO: {
#if 0
      mult_pym_buf = reinterpret_cast<iot_mult_img_info_t *>(buf);
      ret = HB_VPS_GetChnFrame(grp_id, pym_chn, &out_pym_buf, timeout);
      if (ret != 0) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " HB_VPS_GetChnFrame error, ret: " << ret;
        return ret;
      }
      mult_pym_buf->img_info[mult_pym_buf->img_num] = out_pym_buf;
      mult_pym_buf->img_num++;
#endif
      break;
    }
    case IOT_VIO_FEEDBACK_SRC_INFO: {
      fb_src_buf = reinterpret_cast<hb_vio_buffer_t*>(buf);
      if (fb_queue_.size() == 0) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " fb_queue size is zero, pipe_id: " << pipe_id_;
        return -1;
      }
      vb_buf = fb_queue_.front();
      fb_queue_.pop();
      if (vb_buf == NULL) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " vb buf is nullptr";
        return -1;
      }
      memcpy(fb_src_buf, vb_buf, sizeof(hb_vio_buffer_t));
      fb_queue_.push(vb_buf);  // push again for cycle queue
      break;
    }
    default: {
      LOGE << "info type: " << info_type << " error";
      return -1;
    }
  }

  return 0;
}

int VpsModule::VpsSetInfo(int info_type, void *buf) {
  int ret = -1;
  int timeout, grp_id;
  hb_vio_buffer_t *fb_buf = nullptr;

  grp_id = pipe_id_;
  switch (info_type) {
  case IOT_VIO_FEEDBACK_FLUSH:
    ret = 0;
    break;
  case IOT_VIO_FEEDBACK_PYM_PROCESS:
    fb_buf = reinterpret_cast<hb_vio_buffer_t*>(buf);
    timeout = vio_cfg_.pym_cfg.timeout;
    /* cache flush */
    if (vb_cache_en_ == 1) {
      uint32_t height = fb_buf->img_addr.height;
      uint32_t stride = fb_buf->img_addr.stride_size;
      uint32_t size_y = stride * height;
      uint32_t size_uv = size_y / 2;
      char *vir_ptr0 = fb_buf->img_addr.addr[0];
      uint32_t phy_ptr0 = fb_buf->img_addr.paddr[0];
      char *vir_ptr1 = fb_buf->img_addr.addr[1];
      uint32_t phy_ptr1 = fb_buf->img_addr.paddr[1];

      if (vir_ptr0) {
        ret = HB_SYS_CacheFlush(phy_ptr0, vir_ptr0, size_y);
        if (ret) {
          LOGE << "feedback pym process, cache flush y failed,"
            << " ret:" << ret;
          return ret;
        }
      }
      if (vir_ptr1) {
        ret = HB_SYS_CacheFlush(phy_ptr1, vir_ptr1, size_uv);
        if (ret) {
          LOGE << "feedback pym process, cache flush uv failed,"
            << " ret:" << ret;
          return ret;
        }
      }
    }

    /* send frame to vps module */
    ret = HB_VPS_SendFrame(grp_id, fb_buf, timeout);
    if (ret) {
      LOGE << "info_type: " << kIotTypeInfo[info_type]
        << " HB_VPS_SendFrame error, ret: " << ret;
      return ret;
    }
    break;
  default:
    LOGE << "info type: " << info_type << " error";
    return -1;
  }

  return 0;
}

void VpsModule::HbAllocPymBuffer() {
  std::lock_guard<std::mutex> lg(mutex_);
  consumed_pym_buffers_++;
}

void VpsModule::HbFreePymBuffer() {
  std::lock_guard<std::mutex> lg(mutex_);
  if (0 == consumed_pym_buffers_)
    return;
  consumed_pym_buffers_--;
}

int VpsModule::HbManagerPymBuffer(int max_buf_num) {
  std::lock_guard<std::mutex> lg(mutex_);
  int ret = -1;
  int queue_size, total_buf_num;

  /* queue_size = RingQueue<IotPymInfo>::Instance().Size(); */
  queue_size = pym_rq_->Size();
  total_buf_num = queue_size + consumed_pym_buffers_;
  pr_info("pipe_id:%d max_buf_num:%d queue_size:%d"
      " consumed_pym_buffers:%d total_buf_num:%d\n",
      pipe_id_, max_buf_num, queue_size,
      consumed_pym_buffers_, total_buf_num);
  if (total_buf_num >= max_buf_num) {
  /**
   * Attention:
   * 1. if consumed_pym_buffers is more than the (max_buf_num - 1),
   *    vpsmodule will not discard this pym frame.
   * 2. if queue_size is more than the max_buf_num,
   *    vpsmodule will discard this pym frame.
   */
    LOGW  << "pipe_id: " << pipe_id_
      << " PYM Data is Full!"
      << " total_buf_num: " << total_buf_num
      << " queue_size: " << queue_size
      << " consumed_pym_buffers: " << consumed_pym_buffers_
      << " max_buf_num: " << max_buf_num;
    if (queue_size >= max_buf_num) {
      LOGW << "pipe_id: " << pipe_id_
        << " PYM RingQueue will discard first frame data!";
      ret = sem_trywait(&pym_sem_);
      if (ret) {
        LOGE << "sem try wait failure, ret: " << ret;
        return ret;
      }
      pym_rq_->DeleteOne();
    }
  }
  return 0;
}

int VpsModule::VpsFreeInfo(int info_type, void *buf) {
  int ret = -1;
  int grp_id = pipe_id_;
  int ipu_chn_num, pym_chn_num;
  int ipu_valid_chn, pym_valid_chn;

  switch (info_type) {
  case IOT_VIO_SRC_INFO: {
    ipu_chn_num = vio_chn_info_.ipu_chn_num;
    if (ipu_chn_num <= 0 || ipu_chn_num >= MAX_CHN_NUM) {
      LOGE << "info_type: " << kIotTypeInfo[info_type]
        << " ipu chn num is invalid: " << ipu_chn_num
        << " grp_id: " << grp_id;
      return -1;
    }

    for (int i = 0; i < ipu_chn_num; i++) {
      ipu_valid_chn = vio_chn_info_.ipu_valid_chn[i];
      ret = HB_VPS_ReleaseChnFrame(grp_id, ipu_valid_chn, buf);
      if (ret != 0) {
        LOGE << "info_type: " << kIotTypeInfo[info_type]
          << " HB_VPS_ReleaseChnFrame error, ret: " << ret;
        return ret;
      }
      break;
    }
    break;
  }
  case IOT_VIO_PYM_INFO:
    pym_chn_num = vio_chn_info_.pym_chn_num;
    pym_valid_chn = vio_chn_info_.pym_valid_chn[0];
    if (pym_chn_num != 1) {
      LOGE << "info_type: " << kIotTypeInfo[info_type]
        << " pym chn num is invalid: " << pym_chn_num
        << " grp_id: " << grp_id;
      return -1;
    }
    ret = HB_VPS_ReleaseChnFrame(grp_id, pym_valid_chn, buf);
    if (ret != 0) {
      LOGE << "info_type: " << kIotTypeInfo[info_type]
        << " HB_VPS_ReleaseChnFrame error, ret: " << ret;
      return ret;
    }
    HbFreePymBuffer();
    break;
  case IOT_VIO_FEEDBACK_SRC_INFO:
    ret = 0;
    break;
#if 0
  case IOT_VIO_PYM_MULT_INFO:
    ret = HB_VPS_ReleaseChnFrame(grp_id, pym_chn, buf);
    if (ret != 0) {
      LOGE << "info_type: " << kIotTypeInfo[info_type]
        << " HB_VPS_ReleaseChnFrame error, ret: " << ret;
    }
    break;
#endif
  default:
    LOGE << "info type: " << info_type << " error";
    return -1;
  }

  return 0;
}

void VpsModule::HbGetPymDataThread() {
  // start get pym data
  int ret = -1;
  int grp_id = pipe_id_;
  int index = 0;
  uint64_t ts_diff, ts_cur, ts_pre;
  IotPymInfo pym_info;
  int frame_id_diff, frame_id, frame_id_pre;
  int pym_chn = vio_chn_info_.pym_valid_chn[0];
  int timeout = vio_cfg_.pym_cfg.timeout;
  int frame_depth = vio_cfg_.pym_cfg.frameDepth;
  int max_pym_buffer = frame_depth;
  int frame_count = 0;

  LOGI << "Enter Get PYM Data Thread, pipe_id: " << pipe_id_
    << " start_flag: " << pym_start_flag_
    << " max_pym_buffer is: " << max_pym_buffer;
  pym_rq_ = std::make_shared<VioRingQueue<IotPymInfo>>();
  pym_rq_->Init(max_pym_buffer, [](IotPymInfo &elem) {
      int ret;
      ret = HB_VPS_ReleaseChnFrame(elem.grp_id, elem.pym_chn, &elem.pym_buf);
      if (ret) {
      LOGE << "HB_VPS_ReleaseChnFrame error, ret: " << ret;
      }
      });

  auto start_time = std::chrono::system_clock::now();
  while (pym_start_flag_) {
    memset(&pym_info, 0, sizeof(IotPymInfo));

    ret = HbManagerPymBuffer(max_pym_buffer);
    if (ret) {
      LOGE << "pipe_id: " << grp_id
        << " HbManagerPymBuffer error, ret: " << ret;
      continue;
    }
    ret = HB_VPS_GetChnFrame(grp_id, pym_chn, &pym_info.pym_buf, timeout);
    if (ret) {
      LOGE << "pipe_id: " << grp_id
        << " HB_VPS_GetChnFrame error, ret: " << ret;
      continue;
    }
    pym_info.grp_id = grp_id;
    pym_info.pym_chn = pym_chn;
    pym_info.buf_index = (index++) % frame_depth;

    /* calculate vio fps */
    {
      frame_count++;
      auto curr_time = std::chrono::system_clock::now();
      auto cost_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            curr_time - start_time);
      if (cost_time.count() >= 1000) {
        LOGW << "pipe_id:" << grp_id << " vio_fps:" << frame_count;
        start_time = std::chrono::system_clock::now();
        frame_count = 0;
      }
    }

    /* print pym info */
    {
      /* cur pym info */
      ts_cur = pym_info.pym_buf.pym_img_info.time_stamp;
      frame_id = pym_info.pym_buf.pym_img_info.frame_id;
      /* pre pym info */
      ts_pre = vio_pym_info_.pym_buf.pym_img_info.time_stamp;
      frame_id_pre = vio_pym_info_.pym_buf.pym_img_info.frame_id;
      /* pym frame ts diff value */
      ts_diff = ts_cur - ts_pre;
      frame_id_diff = frame_id - frame_id_pre;
      pr_warn("pipe_id:%d buf_index[%d] pym_chn:%d "
          "frame_id:%d, frame_id_pre:%d "
          "frame_id_diff:%d ts_diff:%lu "
          "width:%d height:%d\n",
          pym_info.grp_id,
          pym_info.buf_index,
          pym_info.pym_chn,
          frame_id, frame_id_pre, frame_id_diff, ts_diff,
          pym_info.pym_buf.pym[0].width,
          pym_info.pym_buf.pym[0].height);
    }
    /* store last pym info */
    vio_pym_info_ = pym_info;
    pym_rq_->Push(pym_info);
    sem_post(&pym_sem_);
  }
  pym_rq_->Clear();
  LOGD << "Quit get pym data thread, pipe_id: " << pipe_id_;
}

void VpsModule::HbGetFbDataThread() {
  // start get feedback src data
  int ret = -1;
  int pipe_id = pipe_id_;
  int timeout = 2000;
  int src_chn = 0;
  IotVioInfo vio_info;
  int fb_src_type = vio_cfg_.vps_cfg.fb_info.inner_buf_type;
  int fb_src_pipe_id = vio_cfg_.vps_cfg.fb_info.bind_pipe_id;

  LOGI << "Enter Get Feedback Data Thread, pipe_id: " << pipe_id_
    << " start_flag: " << fb_start_flag_
    << " fb_src_type: " << fb_src_type
    << " fb_src_pipe_id: " << fb_src_pipe_id;

  while (fb_start_flag_) {
    memset(&vio_info, 0, sizeof(IotVioInfo));

    // 1.get inner source data
    if (fb_src_type == kSIF_BUF_SOURCE) {
      ret = HB_VIN_GetDevFrame(fb_src_pipe_id, src_chn, &vio_info.vio_buf,
          timeout);
      if (ret) {
        LOGE << "fb_src_pipe_id: " << fb_src_pipe_id
          << " HB_VIN_GetDevFrame error, ret: " << ret;
        continue;
      }
    } else if (fb_src_type == kISP_BUF_SOURCE) {
      LOGW << "isp buffer source is not support...";
      fb_start_flag_ = false;
      continue;
    } else {
      LOGE << "error feedback source type!!!";
      fb_start_flag_ = false;
      continue;
    }

    // 2.send inner source data
    ret = HB_VPS_SendFrame(pipe_id, &vio_info.vio_buf, timeout);
    if (ret) {
      LOGE << "pipe_id: " << pipe_id
        << " HB_VPS_SendFrame error, ret: " << ret;
    }

    // 3.free inner source data
    if (fb_src_type == kSIF_BUF_SOURCE) {
      ret = HB_VIN_ReleaseDevFrame(fb_src_pipe_id, src_chn, &vio_info.vio_buf);
      if (ret) {
        LOGE << "pipe_id: " << pipe_id
          << " HB_VIN_ReleaseDevFrame error, ret: " << ret;
        continue;
      }
    }
  }

  LOGD << "Quit get feedback data thread, pipe_id: " << pipe_id_;
}

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
