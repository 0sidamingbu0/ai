/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */
#ifndef INCLUDE_MULTISOURCESMARTPLUGIN_CONVERT_H_
#define INCLUDE_MULTISOURCESMARTPLUGIN_CONVERT_H_

#include <string>

#include "hobotxsdk/xstream_data.h"
#include "xproto_msgtype/gdcplugin_data.h"
#include "xproto_msgtype/vioplugin_data.h"

using horizon::vision::xproto::basic_msgtype::VioMessage;
using horizon::vision::xproto::basic_msgtype::IpmImageMessage;

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {
using horizon::vision::xproto::basic_msgtype::VioMessage;

class Convertor {
 public:
  /**
   * @brief convert input VioMessage to xstream inputdata.
   * @param input input VioMessage
   * @param input idx multi-images index
   * @param input image input name in xstream workflow config
   * @return xstream::InputDataPtr xstream input
   */
  static xstream::InputDataPtr ConvertInput(const IpmImageMessage *input,
                                            uint32_t idx,
                                            std::string input_name = "image");
  static xstream::InputDataPtr ConvertInput(const VioMessage *input,
                                            std::string input_name = "image");
  static xstream::InputDataPtr ConvertInput(const VioMessage *input,
                                            uint32_t image_idx,
                                            std::string input_name = "image");
};

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  //  INCLUDE_MULTISOURCESMARTPLUGIN_CONVERT_H_
