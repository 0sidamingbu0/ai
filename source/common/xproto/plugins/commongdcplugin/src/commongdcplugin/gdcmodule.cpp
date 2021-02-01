/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2021-01-28
 * @Version: v1.0.0
 * @Brief: IOT GDC Module for Horizon VIO System.
 */

#include "commongdcplugin/gdcmodule.h"
#include <unistd.h>
#include <stdlib.h>
#include <cstdint>
#include <memory>
#include <fstream>
#include "hobotlog/hobotlog.hpp"
namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

GdcModule::GdcModule(std::shared_ptr<GdcConfig> &config) {
  gdc_cfg_ = config;
  gdc_input_w_ = 1280;
  gdc_input_h_ = 720;
  gdc_output_w_ = 256;
  gdc_output_h_ = 512;
  gdc_frame_depth_ = 3;
  ipu_chn_ = 1;
  pym_chn_ = 6;
}

int GdcModule::GetGdcData(const char *gdc_name,
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

int GdcModule::CreateGrp(int grp_id, int grp_w, int grp_h,
    int grp_depth) {
  int ret = -1;
  VPS_GRP_ATTR_S grp_attr;

  memset(&grp_attr, 0, sizeof(VPS_GRP_ATTR_S));
  grp_attr.maxW = grp_w;
  grp_attr.maxH = grp_h;
  grp_attr.frameDepth = grp_depth;
  LOGI << "vps create group, grp_id:" << grp_id
    << " grp_w: " << grp_w << " grp_h: " << grp_h
    << " grp_depth: " << grp_depth;
  ret = HB_VPS_CreateGrp(grp_id, &grp_attr);
  if (ret) {
    LOGE << "HB_VPS_CreateGrp error!!!";
    return ret;
  } else {
    LOGI << "created a group ok: GrpId: " << grp_id;
  }

  return 0;
}

int GdcModule::SetIpuChnAttr(int grp_id, int ipu_chn, int chn_w,
    int chn_h, int chn_depth) {
  int ret = -1;
  VPS_CHN_ATTR_S chn_attr = { 0 };
  uint8_t scale_en = 1;
  chn_attr.enScale = scale_en;
  chn_attr.width = chn_w;
  chn_attr.height = chn_h;
  chn_attr.frameDepth = chn_depth;
  ret = HB_VPS_SetChnAttr(grp_id, ipu_chn, &chn_attr);
  if (ret) {
    LOGE << "HB_VPS_SetChnAttr error!!!";
    return ret;
  } else {
    LOGI << "set ipu chn Attr ok: GrpId: " << grp_id
      << " ipu_chn: " << ipu_chn;
  }
  ret = HB_VPS_EnableChn(grp_id, ipu_chn);
  if (ret < 0) {
    LOGE << "gdc grp enable chn failed!, grp_id: " << grp_id;
    return ret;
  }

  return 0;
}

int GdcModule::SetPymAttr(int grp_id, int pym_chn) {
  int ret = -1;
  VPS_PYM_CHN_ATTR_S pym_chn_attr = { 0 };

  pym_chn_attr.frame_id = 1;
  pym_chn_attr.ds_layer_en = 4;
  pym_chn_attr.frameDepth = 3;
  pym_chn_attr.timeout = 2000;

  ret = HB_VPS_SetPymChnAttr(grp_id, pym_chn, &pym_chn_attr);
  if (ret) {
    LOGE << "HB_VPS_SetPymChnAttr error!!!";
    return ret;
  } else {
    LOGI << "vps set pym chn Attr ok: GrpId: " << grp_id
      << " pym_chn: " << pym_chn;
  }
  HB_VPS_EnableChn(grp_id, pym_chn);

  return 0;
}

int GdcModule::Init() {
  int ret = -1;
  LOGI << "Enter gdc module init";
  if (gdc_cfg_ == nullptr) {
    LOGE << "gdc_cfg_ is nullptr";
    return -1;
  }
  auto chn_num = gdc_cfg_->data_source_num;
  LOGI << "Enter gdc module init, chn_num: " << chn_num;

  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    int grp_id = chn_index + GDC_GROUP_ID;
    const char *gdc_path = gdc_cfg_->gdc_file_path[chn_index].c_str();
    char *gdc_data = nullptr;
    int gdc_len = 0;
    ret = CreateGrp(grp_id, gdc_input_w_, gdc_input_h_, gdc_frame_depth_);
    if (ret) {
      LOGE << "Gdc createGrp failed, grp_id: " << grp_id;
      return ret;
    }
    LOGI << "grp_id: " << grp_id << " gdc_path: " << gdc_path;
    ret = GetGdcData(gdc_path, &gdc_data, &gdc_len);
    if (ret) {
      LOGE << "get gdc data failed, ret: " << ret;
      return ret;
    }
    ROTATION_E gdc_rotate = static_cast<ROTATION_E>(0);
    ret = HB_VPS_SetGrpGdc(grp_id, gdc_data, gdc_len, gdc_rotate);
    if (ret) {
      LOGE << "HB_VPS_SetGrpGdc error!!!";
    } else {
      LOGI << "HB_VPS_SetGrpGdc ok: gdc grp_id = " << grp_id;
    }
    ret = HB_VPS_UpdateGdcSize(grp_id, 0, gdc_output_w_, gdc_output_h_);
    if (ret) {
      LOGE << "HB_VPS_UpdateGdcSize error!!!";
    } else {
      LOGI << "HB_VPS_UpdateGdcSize ok: gdc grp_id = " << grp_id;
    }
#if 1
    ret = SetIpuChnAttr(grp_id, ipu_chn_, gdc_output_w_,
        gdc_output_h_, gdc_frame_depth_);
    if (ret) {
      LOGE << "Gdc set ipu chn attr failed, grp_id: " << grp_id
        << " ipu_chn: " << ipu_chn_;
      return ret;
    }
#endif
#if 1
    ret = SetIpuChnAttr(grp_id, pym_chn_, gdc_output_w_,
        gdc_output_h_, gdc_frame_depth_);
    if (ret) {
      LOGE << "Gdc set ipu chn attr failed, grp_id: " << grp_id
        << " ipu_chn_pym: " << pym_chn_;
      return ret;
    }
    ret = SetPymAttr(grp_id, pym_chn_);
    if (ret) {
      LOGE << "Gdc set ipu chn attr failed, grp_id: " << grp_id
        << " pym_chn: " << pym_chn_;
      return ret;
    }
#endif
  }

  return 0;
}

int GdcModule::DeInit() {
  int ret = -1;
  auto chn_num = gdc_cfg_->data_source_num;

  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    int grp_id = chn_index + GDC_GROUP_ID;
    ret = HB_VPS_DestroyGrp(grp_id);
    if (ret < 0) {
      LOGE << "gdc grp deinit failed!, grp_id: " << grp_id;
      return ret;
    }
  }

  return 0;
}

int GdcModule::Start() {
  int ret = -1;
  auto chn_num = gdc_cfg_->data_source_num;

  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    int grp_id = chn_index + GDC_GROUP_ID;
    ret = HB_VPS_StartGrp(grp_id);
    if (ret < 0) {
      LOGE << "gdc grp start failed!, grp_id: " << grp_id;
      return ret;
    }
  }

  return 0;
}

int GdcModule::Stop() {
  int ret = -1;
  auto chn_num = gdc_cfg_->data_source_num;

  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    int grp_id = chn_index + GDC_GROUP_ID;
    ret = HB_VPS_StopGrp(grp_id);
    if (ret < 0) {
      LOGE << "gdc grp stop failed!, grp_id: " << grp_id;
      return ret;
    }
    ret = HB_VPS_DisableChn(grp_id, 0);
    if (ret < 0) {
      LOGE << "gdc grp enable chn failed!, grp_id: " << grp_id;
      return ret;
    }
  }

  return 0;
}

void *GdcModule::CreateSrcAddrInfo() {
  auto *pvio_image = reinterpret_cast<void*>(
      std::calloc(1, sizeof(hb_vio_buffer_t)));
  if (nullptr == pvio_image) {
    LOGE << "std::calloc vio buffer failed";
    return nullptr;
  }

  return pvio_image;
}

int GdcModule::ConvertSrcInfo(void *src_buf,
    std::shared_ptr<SrcImageFrame> &src_img) {
  auto src_buffer = static_cast<hb_vio_buffer_t*>(src_buf);
  if (nullptr == src_buffer || nullptr == src_img) {
    LOGE << "src_buffer or src_img is nullptr"
      << " src_buffer: " << src_buffer
      << " src_img: " << src_img;
    return -1;
  }
  src_img->src_info.width = src_buffer->img_addr.width;
  src_img->src_info.height = src_buffer->img_addr.height;
  src_img->src_info.stride = src_buffer->img_addr.stride_size;
  src_img->src_info.y_paddr = src_buffer->img_addr.paddr[0];
  src_img->src_info.c_paddr = src_buffer->img_addr.paddr[1];
  src_img->src_info.y_vaddr =
    reinterpret_cast<uint64_t>(src_buffer->img_addr.addr[0]);
  src_img->src_info.c_vaddr =
    reinterpret_cast<uint64_t>(src_buffer->img_addr.addr[1]);

  return 0;
}

int GdcModule::dump_nv12_file(std::string &filename,
    char *y_addr, char *uv_addr,
    uint32_t y_size, uint32_t uv_size) {
  FILE *yuvFd = NULL;
  char *buffer = NULL;

  yuvFd = fopen(filename.c_str(), "w+");
  if (yuvFd == NULL) {
    std::cout << "open file: " << filename << " failed" << std::endl;
    return -1;
  }

  std::cout << "filename: " << filename << " yuvfd: " << yuvFd << std::endl;
  std::cout << "y_addr: " << reinterpret_cast<void*>(y_addr) << std::endl;
  std::cout << "uv_addr: " << reinterpret_cast<void*>(uv_addr) << std::endl;
  std::cout << "y_size: " << static_cast<int>(y_size) << std::endl;

  buffer = reinterpret_cast<char*>(malloc(y_size + uv_size));
  if (buffer == NULL) {
    std::cout << "malloc failed " << std::endl;
    fclose(yuvFd);
    return -1;
  }

  memcpy(buffer, y_addr, y_size);
  memcpy(buffer + y_size, uv_addr, uv_size);

  fflush(stdout);

  auto total_size = y_size + uv_size;
  auto act_size = fwrite(buffer, 1, total_size, yuvFd);
  if (act_size != total_size) {
    std::cout << "fwrite total size: " << total_size
      << " actual_size: " << act_size
      << " failed" << std::endl;
    fclose(yuvFd);
    return -1;
  }

  fflush(yuvFd);
  fclose(yuvFd);
  std::cout << "dump " << filename << " success..." << std::endl;

  return 0;
}

int GdcModule::Input(std::shared_ptr<PymImageFrame> &pym_img,
    int chn_id) {
  int ret = -1;
  int pym_layer = gdc_cfg_->pym_layer;
  hb_vio_buffer_t src_buf = { 0 };
  int grp_id = chn_id + GDC_GROUP_ID;
  int timeout = 2000;
  if (nullptr == pym_img) {
    LOGE << "pym_img is nullptr";
    return -1;
  }
  uint64_t frame_id = pym_img->frame_id;

  ImageLevelInfo &imageinfo = pym_img->down_scale[pym_layer];
  src_buf.img_info.planeCount = 2;
  src_buf.img_info.img_format = 8;
  src_buf.img_addr.width = imageinfo.width;
  src_buf.img_addr.height = imageinfo.height;
  src_buf.img_addr.stride_size = imageinfo.stride;
  src_buf.img_addr.addr[0] =
    reinterpret_cast<char*>(imageinfo.y_vaddr);
  src_buf.img_addr.addr[1] =
    reinterpret_cast<char*>(imageinfo.c_vaddr);
  src_buf.img_addr.paddr[0] = imageinfo.y_paddr;
  src_buf.img_addr.paddr[1] = imageinfo.c_paddr;

  if ((access("gdc_input.txt", F_OK)) == 0) {
    std::string filename = std::to_string(frame_id) + "_" +
      "chn" + std::to_string(chn_id) + "_gdc_input_pym" +
      std::to_string(pym_layer) + ".yuv";
    char *y_addr = src_buf.img_addr.addr[0];
    char *uv_addr = src_buf.img_addr.addr[1];
    uint32_t y_size = (uint32_t)(imageinfo.stride * imageinfo.height);
    uint32_t uv_size = y_size / 2;
    LOGI << "dump gdc input filename: " << filename
      << " width: " << imageinfo.width
      << " height: " << imageinfo.height
      << " stride: " << imageinfo.stride;
    dump_nv12_file(filename, y_addr, uv_addr, y_size, uv_size);
  }

  ret = HB_VPS_SendFrame(grp_id, &src_buf, timeout);
  if (ret) {
    LOGE << "Gdc module input: "
      << " HB_VPS_SendFrame error, ret: " << ret
      << " grp_id: " << grp_id;
    return ret;
  }

  return 0;
}

int GdcModule::Output(int chn_id, int frame_id, void *gdc_buf) {
  if (gdc_buf == nullptr) {
    LOGE << "gdc_buf is nulptr!";
    return -1;
  }
  int ret = -1;
  int pym_layer = gdc_cfg_->pym_layer;
  int grp_id = chn_id + GDC_GROUP_ID;
  int timeout = 2000;

  ret = HB_VPS_GetChnFrame(grp_id, ipu_chn_, gdc_buf, timeout);
  if (ret != 0) {
    LOGE << "Gdc module output: "
      << " HB_VPS_GetChnFrame error, ret: " << ret
      << " grp_id: " << grp_id;
    return ret;
  }
  if ((access("gdc_output.txt", F_OK)) == 0) {
    hb_vio_buffer_t *src_buf = static_cast<hb_vio_buffer_t*>(gdc_buf);
    std::string filename = std::to_string(frame_id) + "_" +
      "chn" + std::to_string(chn_id) + "_gdc_output_pym" +
      std::to_string(pym_layer) + ".yuv";
    char *y_addr = src_buf->img_addr.addr[0];
    char *uv_addr = src_buf->img_addr.addr[1];
    uint32_t width = (uint32_t)src_buf->img_addr.width;
    uint32_t stride = (uint32_t)src_buf->img_addr.stride_size;
    uint32_t height = (uint32_t)src_buf->img_addr.height;
    uint32_t y_size = (uint32_t)(stride * height);
    uint32_t uv_size = y_size / 2;
    LOGI << "dump gdc output filename: " << filename
      << " width: " << width
      << " height: " << height
      << " stride: " << stride;
    dump_nv12_file(filename, y_addr, uv_addr, y_size, uv_size);
  }

#if 0
  ret = HB_VPS_ReleaseChnFrame(grp_id, ipu_chn_, gdc_buf);
  if (ret != 0) {
    LOGE << "Gdc module output: "
      << " HB_VPS_ReleaseChnFrame error, ret: " << ret
      << " grp_id: " << grp_id;
    return ret;
  }
#endif
  return 0;
}

int GdcModule::FreeSrcImage(int chn_id, void *gdc_buf) {
  int ret = -1;
  int grp_id = chn_id + GDC_GROUP_ID;

  ret = HB_VPS_ReleaseChnFrame(grp_id, ipu_chn_, gdc_buf);
  if (ret != 0) {
    LOGE << "Gdc module output: "
      << " HB_VPS_ReleaseChnFrame error, ret: " << ret
      << " grp_id: " << grp_id;
    return ret;
  }

  return 0;
}

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
