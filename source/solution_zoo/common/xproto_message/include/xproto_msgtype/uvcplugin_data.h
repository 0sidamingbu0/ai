/*
 * @Description: implement of uvcplugin.h
 * @Author: yingmin.li@horizon.ai
 * @Date: 2019-08-24 11:29:24
 * @LastEditors: hao.tian@horizon.ai
 * @LastEditTime: 2019-10-16 16:11:08
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#ifndef XPROTO_MSGTYPE_UVCPLUGIN_DATA_H_
#define XPROTO_MSGTYPE_UVCPLUGIN_DATA_H_

#include <string>
#include <utility>
#include "xstream/vision_type/include/horizon/vision_type/vision_type_util.h"
#include "xproto/message/pluginflow/flowmsg.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace basic_msgtype {

#define TYPE_UVC_MESSAGE "XPLUGIN_UVC_MESSAGE"
#define TYPE_TRANSPORT_MESSAGE "XPLUGIN_TRANSPORT_MESSAGE"

struct UvcMessage : XProtoMessage {
  UvcMessage() { type_ = TYPE_UVC_MESSAGE; }
  std::string Serialize() override { return "Default uvc message"; }
  virtual ~UvcMessage() = default;
};

struct TransportMessage : XProtoMessage {
  explicit TransportMessage(const std::string& proto) {
    //  这里并不会起到预期右值转移的作用，
    //  因为const限制了右值转移函数的调用
    proto_ = std::move(proto);
    type_ = TYPE_TRANSPORT_MESSAGE;
  }
  explicit TransportMessage(std::string&& proto) {
    proto_ = std::forward<std::string>(proto);
    type_ = TYPE_TRANSPORT_MESSAGE;
  }
  std::string Serialize() override { return "Default transport message"; }
  virtual ~TransportMessage() = default;

  std::string proto_;
};
#ifdef USE_MC
#define TYPE_APIMAGE_MESSAGE "XPLUGIN_APIMAGE_MESSAGE"
#define TYPE_MC_UPSTREAM_MESSAGE "XPLUGIN_MC_UPSTREAM_MESSAGE"
struct APImageMessage : XProtoMessage {
  explicit APImageMessage(const std::string& buff_str,
                          const std::string& img_type,
                          uint32_t width, uint32_t height,
                          uint64_t seq_id) : sequence_id_(seq_id) {
    HorizonVisionAllocImage(&img_);
    if (img_type == "JPEG") {
      img_->pixel_format = kHorizonVisionPixelFormatImageContainer;
    } else if (img_type == "NV12") {
      img_->pixel_format = kHorizonVisionPixelFormatRawNV12;
    } else if (img_type == "BGR") {
      img_->pixel_format = kHorizonVisionPixelFormatRawBGR;
    } else {
      HOBOT_CHECK(false) << "unsupport AP image type";
    }
    img_->data_size = buff_str.size();
    img_->data = reinterpret_cast<uint8_t *>(
        HorizonVisionMemDup(buff_str.c_str(), buff_str.size()));
    type_ = TYPE_APIMAGE_MESSAGE;
  }

  std::string Serialize() override { return "APImageMessage"; }
  virtual ~APImageMessage() {
    HorizonVisionFreeImage(img_);
    img_ = nullptr;
  }

  HorizonVisionImage *img_ = nullptr;
  uint64_t sequence_id_ = 0;
  std::string img_name;
};

struct APRespMessage : public XProtoMessage{
  explicit APRespMessage(uint64_t seq_id) :
      sequence_id_(seq_id) {
    type_ = TYPE_MC_UPSTREAM_MESSAGE;
  }
  std::string Serialize() { return proto_; }
  uint64_t sequence_id_ = 0;
  std::string proto_;
};

#endif

}  // namespace basic_msgtype
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // UVCPLUGIN_INCLUDE_UVCPLUGIN_UVCPLUGIN_H_
