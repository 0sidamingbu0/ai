/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-08-12
 * @Version: v1.0.0
 * @Brief: IOT VIO Pipe Manager for Horizon VIO System.
 */

#include <string.h>
#include <mutex>
#include "iotviomanager/viopipemanager.h"
#include "iotviomanager/vio_data_type.h"
#include "hobotlog/hobotlog.hpp"
#include "./hb_vp_api.h"
#include "./hb_vio_interface.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

int VioPipeManager::GetPipeId(const std::vector<std::string> &cfg_file) {
  std::lock_guard<std::mutex> lg(mutex_);
  int pipe_id_tmp;
  HOBOT_CHECK(pipe_id_ < MAX_PIPE_NUM) << "get pipe_id failed";
  pipe_id_tmp = pipe_id_;
  LOGI << "get vio pipe_id value: " << pipe_id_;
  pipe_id_ = pipe_id_ + cfg_file.size();

  return pipe_id_tmp;
}

int VioPipeManager::Reset() {
  std::lock_guard<std::mutex> lg(mutex_);
  pipe_id_ = 0;

  LOGD << "viopipemanager reset pipe_id success...";
  return 0;
}

int VioPipeManager::VbInit() {
  std::lock_guard<std::mutex> lg(mutex_);
  int ret = -1;
  VP_CONFIG_S vp_config;

  vb_init_ref_cnt_++;
  if (vb_init_ == true) {
    LOGW << "vp has already init";
    return 0;
  }

  memset(&vp_config, 0x00, sizeof(VP_CONFIG_S));
  vp_config.u32MaxPoolCnt = MAX_POOL_CNT;
  ret = HB_VP_SetConfig(&vp_config);
  if (ret) {
    LOGE << "vp set config failed, ret: " << ret;
    return ret;
  }

  ret = HB_VP_Init();
  if (ret) {
    LOGE << "vp init failed, ret: " << ret;
    return ret;
  }
  vb_init_ = true;

  LOGI << "viopipemanager vb init success...";
  return 0;
}

int VioPipeManager::VbDeInit() {
  std::lock_guard<std::mutex> lg(mutex_);
  int ret = -1;

  vb_init_ref_cnt_--;
  if (vb_init_ref_cnt_ == 0) {
    LOGI << "vb start attemp deinit";
  } else {
    LOGW << "vb init referent cnt: " << vb_init_ref_cnt_;
    return 0;
  }
  if (vb_init_ == false) {
    LOGW <<  "vp has not init!";
    return 0;
  }
  ret = HB_VP_Exit();
  if (ret == 0) {
    LOGI << "vp exit ok!";
  } else {
    LOGE << "vp exit error!";
    return ret;
  }
  vb_init_ = false;

  return 0;
}

int VioPipeManager::AllocVbBuf2Lane(int index, void *buf,
    uint32_t size_y, uint32_t size_uv, int vb_cache_en) {
  std::lock_guard<std::mutex> lg(mutex_);
  int ret = -1;
  hb_vio_buffer_t *buffer = reinterpret_cast<hb_vio_buffer_t*>(buf);

  if (vb_init_ == false) {
    LOGE << "vp has not init";
    return -1;
  }

  if (buffer == nullptr) {
    LOGE << "buffer is nullptr";
    return -1;
  }

  if (vb_cache_en == 1) {
    ret = HB_SYS_AllocCached(&buffer->img_addr.paddr[0],
        reinterpret_cast<void**>(&buffer->img_addr.addr[0]),
        size_y);
    if (ret) {
      LOGE << "AllocVbBuf2Lane index: " << index
        << "alloc size_y cached vb buffer error," << " ret: " << ret;
      return ret;
    }
    ret = HB_SYS_AllocCached(&buffer->img_addr.paddr[1],
        reinterpret_cast<void**>(&buffer->img_addr.addr[1]),
        size_uv);
    if (ret) {
      LOGE << "AllocVbBuf2Lane index: " << index
        << "alloc size_uv cached vb buffer error," << " ret: " << ret;
      return ret;
    }
  } else {
    ret = HB_SYS_Alloc(&buffer->img_addr.paddr[0],
        reinterpret_cast<void**>(&buffer->img_addr.addr[0]),
        size_y);
    if (ret) {
      LOGE << "AllocVbBuf2Lane index: " << index
        << "alloc size_y vb buffer error," << " ret: " << ret;
      return ret;
    }

    ret = HB_SYS_Alloc(&buffer->img_addr.paddr[1],
        reinterpret_cast<void**>(&buffer->img_addr.addr[1]),
        size_uv);
    if (ret) {
      LOGE << "AllocVbBuf2Lane index: " << index
        << "alloc size_uv vb buffer error," << " ret: " << ret;
      return ret;
    }
  }


#ifdef DEBUG
  LOGD << "mmzAlloc index: " << index;
  LOGD << "vio_buf_addr: "   << buffer;
  LOGD << "buf_y_paddr: "    << buffer->img_addr.paddr[0];
  LOGD << "buf_y_vaddr "
       << reinterpret_cast<void*>(buffer->img_addr.addr[0]);
  LOGD << "buf_uv_paddr: "   << buffer->img_addr.paddr[1];
  LOGD << "buf_uv_vaddr: "
       << reinterpret_cast<void*>(buffer->img_addr.addr[1]);
#endif

  return 0;
}

int VioPipeManager::FreeVbBuf2Lane(int index, void *buf) {
  std::lock_guard<std::mutex> lg(mutex_);
  int ret = -1;
  hb_vio_buffer_t *buffer = reinterpret_cast<hb_vio_buffer_t*>(buf);

  if (vb_init_ == false) {
    LOGE << "vp has not init";
    return -1;
  }

  ret = HB_SYS_Free(buffer->img_addr.paddr[0], buffer->img_addr.addr[0]);
  if (ret == 0) {
    if (buffer->img_addr.addr[1] != nullptr) {
      ret = HB_SYS_Free(buffer->img_addr.paddr[1],
          buffer->img_addr.addr[1]);
      if (ret != 0) {
        LOGE << "hb sys free uv vio buf: " << index
          << "failed, ret: " << ret;
        return ret;
      }
    }
  } else {
    LOGE << "hb sys free y vio buf: " << index << "failed, ret: " << ret;
    return ret;
  }

#ifdef DEBUG
  LOGD << "mmzFree index: "  << index;
  LOGD << "vio_buf_addr: "   << buffer;
  LOGD << "buf_y_paddr: "    << buffer->img_addr.paddr[0];
  LOGD << "buf_y_vaddr "     << reinterpret_cast<void*>\
    (buffer->img_addr.addr[0]);
  LOGD << "buf_uv_paddr: "   << buffer->img_addr.paddr[1];
  LOGD << "buf_uv_vaddr: "   << reinterpret_cast<void*>\
    (buffer->img_addr.addr[1]);
#endif
  return 0;
}

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
