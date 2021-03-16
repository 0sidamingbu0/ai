/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_CONVERT_H_
#define INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_CONVERT_H_

#include <string>
#include <vector>
#include "xproto_msgtype/vioplugin_data.h"
#include "xproto_msgtype/smartplugin_data.h"
#include "media_codec/media_codec_manager.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {

using horizon::vision::xproto::basic_msgtype::VioMessage;
using horizon::vision::xproto::basic_msgtype::SmartMessage;

class Convertor {
 public:
  /**
   * @brief get yuv data from VioMessage (only for mono)
   * @param out frame_buf - iot_venc_src_buf_t with yuv type
   * @param in vio_msg
   * @param in layer - number
   * @param in use_vb - if use vb memory or not
   * @return error code
   */
  static int GetYUV(iot_venc_src_buf_t *frame_buf, VioMessage *vio_msg,
                    int level, uint32_t image_idx);
  /**
   * @brief package smart messge to proto
   * @param out data - protobuf serialized string
   * @param in smart_msg
   * @param in type - VisualConfig::SmartType
   * @param in ori_w - origin width of smart result
   * @param in ori_h - origin height of smart result
   * @param in dst_w - dst width of smart result
   * @param in dst_h - dst height of smart result
   * @return error code
   */
  static int PackSmartMsg(std::string &data, SmartMessage *smart_msg,
                          int ori_w, int ori_h, int dst_w, int dst_h);
};

}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  //  INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_CONVERT_H_
