/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include "multisourcesmartplugin/convert.h"

#include <memory>
#include <string>

#include "hobotlog/hobotlog.hpp"
#include "hobotxsdk/xstream_data.h"
#include "multisourcesmartplugin/util.h"
#include "xproto_msgtype/vioplugin_data.h"

using ImageFramePtr = std::shared_ptr<hobot::vision::ImageFrame>;
namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {
xstream::InputDataPtr Convertor::ConvertInput(const VioMessage *input,
                                              std::string input_name) {
  xstream::InputDataPtr inputdata(new xstream::InputData());
  HOBOT_CHECK(input != nullptr && input->num_ > 0 && input->is_valid_uri_);
  for (uint32_t image_index = 0; image_index < 1; ++image_index)
  {
    LOGI << "convertInput index = " << image_index;
    xstream::BaseDataPtr xstream_input_data;
    if (input->image_.size() > image_index) {
      std::shared_ptr<hobot::vision::PymImageFrame> pym_img =
          input->image_[image_index];
      LOGI << "vio message, frame_id = " << pym_img->frame_id;
      auto xstream_img =
          std::make_shared<xstream::XStreamData<ImageFramePtr>>();
      xstream_img->type_ = "ImageFrame";
      xstream_img->value =
          std::static_pointer_cast<hobot::vision::ImageFrame>(pym_img);
      LOGI << "Input Frame ID = " << xstream_img->value->frame_id
           << ", Timestamp = " << xstream_img->value->time_stamp
           << ", Channel ID = " << xstream_img->value->channel_id;
      xstream_input_data = xstream::BaseDataPtr(xstream_img);
    } else {
      xstream_input_data = std::make_shared<xstream::BaseData>();
      xstream_input_data->state_ = xstream::DataState::INVALID;
    }

    if (image_index == uint32_t{0}) {
      if (input->num_ == 1) {
        xstream_input_data->name_ = input_name;  // default is image
      } else {
        LOGW << "image input name may has error";
        xstream_input_data->name_ = "rgb_image";
      }
    } else {
      LOGW << "image input name may has error";
      xstream_input_data->name_ = "nir_image";
    }

    LOGI << "input name:" << xstream_input_data->name_;
    inputdata->datas_.emplace_back(xstream_input_data);
  }

  return inputdata;
}

xstream::InputDataPtr Convertor::ConvertInput(const VioMessage *input,
                                              uint32_t image_index,
                                              std::string input_name) {
  xstream::InputDataPtr inputdata(new xstream::InputData());
  HOBOT_CHECK(input != nullptr && input->num_ > image_index &&
              input->is_valid_uri_);
  LOGI << "convertInput index = " << image_index;
  xstream::BaseDataPtr xstream_input_data;
  if (input->image_.size() > image_index) {
    std::shared_ptr<hobot::vision::PymImageFrame> pym_img =
        input->image_[image_index];
    LOGI << "vio message, frame_id = " << pym_img->frame_id;
    auto xstream_img = std::make_shared<xstream::XStreamData<ImageFramePtr>>();
    xstream_img->type_ = "ImageFrame";
    xstream_img->value =
        std::static_pointer_cast<hobot::vision::ImageFrame>(pym_img);
    LOGI << "Input Frame ID = " << xstream_img->value->frame_id
         << ", Timestamp = " << xstream_img->value->time_stamp
         << ", Channel ID = " << xstream_img->value->channel_id;
    xstream_input_data = xstream::BaseDataPtr(xstream_img);
  } else {
    xstream_input_data = std::make_shared<xstream::BaseData>();
    xstream_input_data->state_ = xstream::DataState::INVALID;
  }
  xstream_input_data->name_ = input_name;  // default is image

  LOGI << "input name:" << xstream_input_data->name_;
  inputdata->datas_.emplace_back(xstream_input_data);

  return inputdata;
}

xstream::InputDataPtr Convertor::ConvertInput(const IpmImageMessage *input,
                                              uint32_t idx,
                                              std::string input_name) {
  xstream::InputDataPtr inputdata(new xstream::InputData());
  HOBOT_CHECK(input != nullptr && input->num_ > 0 && input->is_valid_uri_);
  LOGI << "convertInput index = " << idx;
  xstream::BaseDataPtr xstream_input_data;
  if (input->num_ > idx) {
    std::shared_ptr<hobot::vision::CVImageFrame> ipm_img =
      input->ipm_imgs_[idx];
    LOGI << "ipm message, frame_id = " << ipm_img->frame_id;
    auto xstream_img = std::make_shared<xstream::XStreamData<ImageFramePtr>>();
    xstream_img->type_ = "IpmImageFrame";   // TODO(shiyu.fu): check type_
    xstream_img->value =
      std::static_pointer_cast<hobot::vision::ImageFrame>(ipm_img);
    LOGI << "IPM Input Frame ID = " << xstream_img->value->frame_id
        << ", Timestamp = " << xstream_img->value->time_stamp
        << ", Channel ID = " << xstream_img->value->channel_id;
    xstream_input_data = xstream::BaseDataPtr(xstream_img);
  } else {
    xstream_input_data = std::make_shared<xstream::BaseData>();
    xstream_input_data->state_ = xstream::DataState::INVALID;
  }

  xstream_input_data->name_ = input_name;
  LOGI << "input name: " << xstream_input_data->name_;
  inputdata->datas_.emplace_back(xstream_input_data);
  return inputdata;
}

}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
