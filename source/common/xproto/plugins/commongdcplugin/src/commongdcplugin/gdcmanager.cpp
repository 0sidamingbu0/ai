/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2021-01-28
 * @Version: v1.0.0
 * @Brief: IOT GDC Manager for Horizon VIO System.
 */

#include "commongdcplugin/gdcmanager.h"
#include <string>
#include <vector>
#include "hobotlog/hobotlog.hpp"
#include "commongdcplugin/gdcmodule.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

int GdcManager::Init() {
  int ret = -1;
  LOGI << "Enter GdcManager Init";
  if (gdc_cfg_) {
    gdc_enable_ = gdc_cfg_->gdc_enable;
  } else {
    LOGE << "gdc cfg is nullptr";
    return -1;
  }
  if (gdc_enable_) {
    gdc_module_ = std::make_shared<GdcModule>(gdc_cfg_);
    ret = gdc_module_->Init();
    if (ret) {
      LOGE << "gdc module init failed!!!";
      return ret;
    }
  } else {
    LOGW << "gdc module is not enable...";
  }
  LOGI << "GdcManager Init Success...";
  return 0;
}

int GdcManager::DeInit() {
  int ret = -1;
  LOGI << "Enter GdcManager DeInit";
  if (gdc_enable_) {
    ret = gdc_module_->DeInit();
    if (ret) {
      LOGE << "gdc module deinit failed!!!";
      return ret;
    }
  }
  LOGI << "GdcManager DeInit Success...";
  return 0;
}

int GdcManager::Start() {
  int ret = -1;
  LOGI << "Enter GdcManager Start";
  if (gdc_enable_) {
    ret = gdc_module_->Start();
    if (ret) {
      LOGE << "gdc module start failed!!!";
      return ret;
    }
  }
  is_running_ = true;
  LOGI << "GdcManager Start Success...";
  return 0;
}

int GdcManager::Stop() {
  int ret = -1;
  LOGI << "Enter GdcManager Stop";
  if (gdc_enable_) {
    ret = gdc_module_->Stop();
    if (ret) {
      LOGE << "gdc module stop failed!!!";
      return ret;
    }
  }
  is_running_ = false;
  LOGI << "GdcManager Stop Success...";
  return 0;
}

int GdcManager::Input(
    std::shared_ptr<PymImageFrame> &pym_image) {
  if (gdc_enable_ == false ||
      is_running_ == false) {
    LOGW << "gdc_enable: " << gdc_enable_
      << "is_running: " << is_running_;;
    return 0;
  }
  int ret = -1;
  int chn_id = pym_image->channel_id;

  ret = gdc_module_->Input(pym_image, chn_id);
  if (ret) {
    LOGE << "gdc module input falied, ret: " << ret
      << " chn_id: " << chn_id;
    return ret;
  }

  return 0;
}

int GdcManager::MultiInput(
    std::vector<std::shared_ptr<PymImageFrame>> &pym_images) {
  if (gdc_enable_ == false ||
      is_running_ == false) {
    LOGW << "gdc_enable: " << gdc_enable_
      << " is_running: " << is_running_;;
    return 0;
  }
  int ret = -1;
  int chn_num = gdc_cfg_->data_source_num;

  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    auto pym_image = pym_images[chn_index];
    int chn_id = pym_image->channel_id;
    ret = gdc_module_->Input(pym_image, chn_id);
    if (ret) {
      LOGE << "gdc module input falied, ret: " << ret
        << " chn_id: " << chn_id;
      return ret;
    }
  }

  return 0;
}

int GdcManager::Output(
    std::shared_ptr<PymImageFrame> &pym_image,
    std::shared_ptr<SrcImageFrame> &gdc_image) {
  if (gdc_enable_ == false ||
      is_running_ == false) {
    LOGW << "gdc_enable: " << gdc_enable_
      << "is_running: " << is_running_;;
    return 0;
  }
  if (pym_image == nullptr) {
    LOGE << "pym_image is nullptr";
    return -1;
  }
  int ret = -1;
  void *pvio_image = nullptr;

  auto gdc_image_frame_ptr = std::make_shared<SrcImageFrame>();
  pvio_image = gdc_module_->CreateSrcAddrInfo();
  if (pvio_image == nullptr) {
    LOGE << "vps module create src addr failed";
    return -1;
  }
  gdc_image_frame_ptr->channel_id = pym_image->channel_id;
  gdc_image_frame_ptr->time_stamp = pym_image->time_stamp;
  gdc_image_frame_ptr->frame_id = pym_image->frame_id;
  gdc_image_frame_ptr->context = pvio_image;
  ret = gdc_module_->Output(pym_image->channel_id,
      pym_image->frame_id, pvio_image);
  if (ret) {
    LOGE << "gdc module ouput failed, chn_id: " << pym_image->channel_id;
    goto err;
  }
  ret = gdc_module_->ConvertSrcInfo(pvio_image, gdc_image_frame_ptr);
  if (ret) {
    LOGE << "convert src info failed";
    goto err;
  }
  gdc_image = gdc_image_frame_ptr;

  return 0;
err:
  if (pvio_image) {
    std::free(pvio_image);
    pvio_image = nullptr;
  }
  return ret;
}

int GdcManager::MultiOutput(
    std::vector<std::shared_ptr<PymImageFrame>> &pym_images,
    std::vector<std::shared_ptr<SrcImageFrame>> &gdc_images) {
  if (gdc_enable_ == false ||
      is_running_ == false) {
    LOGW << "gdc_enable: " << gdc_enable_
      << " is_running: " << is_running_;;
    return 0;
  }
  int ret = -1;
  int chn_num = gdc_cfg_->data_source_num;
  std::vector<void*> pvio_images;
  void *pvio_image = nullptr;

  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    auto pym_image = pym_images[chn_index];
    if (pym_image == nullptr) {
      LOGE << "pym_image is nullptr, chn_index: " << chn_index;
      return -1;
    }
    auto gdc_image_frame_ptr = std::make_shared<SrcImageFrame>();
    pvio_image = gdc_module_->CreateSrcAddrInfo();
    if (pvio_image == nullptr) {
      LOGE << "vps module create src addr failed, chn_index: " << chn_index;
      goto err;
    }
    pvio_images.push_back(pvio_image);
    gdc_image_frame_ptr->channel_id = pym_image->channel_id;
    gdc_image_frame_ptr->time_stamp = pym_image->time_stamp;
    gdc_image_frame_ptr->frame_id = pym_image->frame_id;
    gdc_image_frame_ptr->context = pvio_image;
    ret = gdc_module_->Output(pym_image->channel_id,
        pym_image->frame_id, pvio_image);
    if (ret) {
      LOGE << "gdc module ouput failed, chn_id: " << pym_image->channel_id;
      goto err;
    }
    ret = gdc_module_->ConvertSrcInfo(pvio_image, gdc_image_frame_ptr);
    if (ret) {
      LOGE << "convert src info failed";
      goto err;
    }
    gdc_images.push_back(gdc_image_frame_ptr);
  }

err:
  for (size_t i = 0; i < pvio_images.size(); i++) {
    pvio_image = pvio_images[i];
    if (pvio_image) {
      std::free(pvio_image);
      pvio_image = nullptr;
    }
  }
  pvio_images.clear();
  return ret;
}

int GdcManager::FreeSrcImage(int chn_id,
    std::vector<std::shared_ptr<SrcImageFrame>> &gdc_images) {
  int ret = -1;
  int image_num = static_cast<int>(gdc_images.size());

  if (image_num <= 0) {
    LOGE << "image num is error!";
    return -1;
  }
  if (image_num == 1) {
    std::shared_ptr<SrcImageFrame> &gdc_image = gdc_images[0];
    void* gdc_buf = gdc_image->context;
    ret = gdc_module_->FreeSrcImage(chn_id, gdc_buf);
    if (ret) {
      LOGE << "free src image failed, ret: " << ret;
      return ret;
    }
    return 0;
  }

  for (int chn_index = 0; chn_index < image_num; chn_index++) {
    std::shared_ptr<SrcImageFrame> &gdc_image = gdc_images[chn_index];
    void* gdc_buf = gdc_image->context;
    ret = gdc_module_->FreeSrcImage(chn_index, gdc_buf);
    if (ret) {
      LOGE << "free src image failed, ret: " << ret;
      return ret;
    }
  }

  return 0;
}

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
