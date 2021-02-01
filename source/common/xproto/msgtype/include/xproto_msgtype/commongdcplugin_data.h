/* @Description: declaration of ipm image message
 * @Author: xudong.du@horizon.ai
 * @Date: 2020-09-08 15:27:06
 * @Last Modified by: xudong.du@horizon.ai
 * @Last Modified time: 2020-09-08 15:54:13
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 * */

#ifndef XPROTO_MSGTYPE_COMMON_GDCPLUGIN_DATA_H_
#define XPROTO_MSGTYPE_COMMON_GDCPLUGIN_DATA_H_

#include <memory>
#include <vector>
#include <string>

#include "horizon/vision_type/vision_error.h"
#include "horizon/vision_type/vision_msg.h"
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/utils/profile.h"
#include "xproto_msgtype/vioplugin_data.h"
#include "hobotlog/hobotlog.hpp"
#include "commongdcplugin/gdcmanager.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace basic_msgtype {
using horizon::vision::xproto::basic_msgtype::VioMessage;
using hobot::vision::SrcImageFrame;
using horizon::vision::xproto::commongdcplugin::GdcManager;

#define TYPE_GDC_MESSAGE "XPLUGIN_GDC_MESSAGE"
struct CustomGdcMessage : VioMessage {
  explicit CustomGdcMessage(
      const std::shared_ptr<GdcManager> &gdc_manager,
      int chn_id,
      std::vector<std::shared_ptr<SrcImageFrame>> &gdc_images) {
    type_ = TYPE_GDC_MESSAGE;
    chn_id_ = chn_id;
    gdc_manager_ = gdc_manager;
    gdc_images_ = gdc_images;
  }
  ~CustomGdcMessage() {
    LOGI << "call ~CustomGdcMessage";
    FreeImage();
  }

  std::string Serialize() override { return "CustomGdcMessage Serialize."; };
  void FreeImage();

 private:
  int chn_id_;
  std::shared_ptr<GdcManager>  gdc_manager_;
  std::vector<std::shared_ptr<SrcImageFrame>> gdc_images_;
};

void CustomGdcMessage ::FreeImage() {
  if (gdc_images_.size() > 0) {
    LOGI << "begin remove gdc image";
    auto res = gdc_manager_->FreeSrcImage(chn_id_, gdc_images_);
    if (res) {
      LOGE << "free gdc info failed";
    }
    gdc_images_.clear();
  }
}

}  // namespace basic_msgtype
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // XPROTO_MSGTYPE_COMMON_GDCPLUGIN_DATA_H_
