/*
 * @Description: implement of vioplugin
 * @Author: fei.cheng@horizon.ai
 * @Date: 2019-08-26 16:17:25
 * @Author: songshan.gong@horizon.ai
 * @Date: 2019-09-26 16:17:25
 * @LastEditors: hao.tian@horizon.ai
 * @LastEditTime: 2019-10-16 16:27:39
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */
#ifndef INCLUDE_VIOMESSAGE_VIOMESSAGE_H_
#define INCLUDE_VIOMESSAGE_VIOMESSAGE_H_

#include <memory>
#include <vector>
#include <string>
#if defined(X3_X2_VIO) || defined(X3_IOT_VIO)
#include "./hb_cam_interface.h"
#include "./hb_vio_interface.h"
#include "./x3_vio_patch.h"
#endif
#if defined(X3_IOT_VIO)
#include "./hb_vio_interface.h"
#endif

#include "hobot_vision/blocking_queue.hpp"

#include "xproto_msgtype/vioplugin_data.h"
#include "iotviomanager/vio_data_type.h"
#include "iotviomanager/viopipeline.h"
#ifdef USE_MC
#include "xproto_msgtype/uvcplugin_data.h"
using horizon::vision::xproto::basic_msgtype::APImageMessage;
#endif
namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {
using hobot::vision::PymImageFrame;
using horizon::vision::xproto::basic_msgtype::VioMessage;

struct ImageVioMessage : VioMessage {
 public:
  ImageVioMessage() = delete;
  explicit ImageVioMessage(
      const std::shared_ptr<VioPipeLine> &vio_pipeline,
      std::vector<std::shared_ptr<PymImageFrame>> &image_frame,
      uint32_t img_num, bool is_valid = true);
  ~ImageVioMessage();

  // serialize proto
  std::string Serialize() { return "No need serialize"; }

  void FreeImage();
  void FreeImage(int tmp);  // 用于释放x3临时回灌功能的接口
  void FreeMultiImage();  // 用于释放x3多路接口

 private:
  std::shared_ptr<VioPipeLine> vio_pipeline_;
};

struct DropVioMessage : VioMessage {
 public:
  DropVioMessage() = delete;
  explicit DropVioMessage(uint64_t timestamp, uint64_t seq_id);
  ~DropVioMessage(){};

  // serialize proto
  std::string Serialize() override;
};

struct DropImageVioMessage : VioMessage {
 public:
  DropImageVioMessage() = delete;
  explicit DropImageVioMessage(
      const std::shared_ptr<VioPipeLine> &vio_pipeline,
      std::vector<std::shared_ptr<PymImageFrame>> &image_frame,
      uint32_t img_num, bool is_valid = true);
  ~DropImageVioMessage();

  // serialize proto
  std::string Serialize() { return "No need serialize"; }

  void FreeImage();
  void FreeMultiImage();  // 用于释放x3多路接口

 private:
  std::shared_ptr<VioPipeLine> vio_pipeline_;
};

struct MultiVioMessage : VioMessage {
 public:
  MultiVioMessage() {
    LOGD << "MultiVioMessage()";
    type_ = TYPE_MULTI_IMAGE_MESSAGE;}
  std::vector<std::shared_ptr<ImageVioMessage>> multi_vio_img_;
  ~MultiVioMessage() {
    LOGD << "~MultiVioMessage";
    multi_vio_img_.clear();}
};

struct ImageFacePicMessage : ImageVioMessage {
 public:
  ImageFacePicMessage() = delete;
  explicit ImageFacePicMessage(
      const std::shared_ptr<VioPipeLine> &vio_pipeline,
      std::vector<std::shared_ptr<PymImageFrame>> &image_frame,
      uint32_t img_num, bool is_valid = true):
      ImageVioMessage(vio_pipeline, image_frame, img_num, is_valid) {
    type_ = TYPE_FACE_PIC_IMAGE_MESSAGE;
  }
  ~ImageFacePicMessage(){}
};


}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif
