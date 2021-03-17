/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */
#include "multisourcewebsocketplugin/convert.h"

#include <string>
#include "multisourcewebsocketplugin/websocketconfig.h"
#include "xproto_msgtype/smartplugin_data.h"
#include "hobotxsdk/xstream_data.h"
#include "horizon/vision/util.h"
#include "hobotlog/hobotlog.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {

int Convertor::GetYUV(iot_venc_src_buf_t *frame_buf, VioMessage *vio_msg,
        int level, uint32_t image_idx) {
  LOGI << "websocketplugin x3 mediacodec: " << __FUNCTION__;
  if (!vio_msg || vio_msg->num_ == 0 || image_idx >= vio_msg->num_)
    return -1;
  auto pym_image = vio_msg->image_[image_idx];
  auto height = pym_image->down_scale[level].height;
  auto width = pym_image->down_scale[level].width;
  auto stride = pym_image->down_scale[level].stride;
  auto y_vaddr = pym_image->down_scale[level].y_vaddr;
  auto y_paddr = pym_image->down_scale[level].y_paddr;
  auto c_vaddr = pym_image->down_scale[level].c_vaddr;
  auto c_paddr = pym_image->down_scale[level].c_paddr;
  HOBOT_CHECK(height) << "width = " << width << ", height = " << height;

  frame_buf->frame_info.width = width;
  frame_buf->frame_info.height = height;
  frame_buf->frame_info.stride = stride;
  frame_buf->frame_info.size = stride * height * 3 / 2;
  frame_buf->frame_info.vir_ptr[0] = reinterpret_cast<char *>(y_vaddr);
  frame_buf->frame_info.phy_ptr[0] = (uint32_t)y_paddr;
  frame_buf->frame_info.vir_ptr[1] = reinterpret_cast<char *>(c_vaddr);
  frame_buf->frame_info.phy_ptr[1] = (uint32_t)c_paddr;
  frame_buf->frame_info.pix_format = HB_PIXEL_FORMAT_NV12;
  return 0;
}

int Convertor::PackSmartMsg(std::string &data, SmartMessage *smart_msg,
                            int ori_w, int ori_h, int dst_w,
                            int dst_h) {
  if (!smart_msg)
    return -1;
  data = smart_msg->Serialize(ori_w, ori_h, dst_w, dst_h);
  return 0;
}

}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
