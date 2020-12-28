/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Songshan Gong
 * @Mail: songshan.gong@horizon.ai
 * @Date: 2019-08-02 03:42:16
 * @Version: v0.0.1
 * @Brief:  convert to xstream inputdata from input VioMessage
 * @Note:  extracted from xperson repo.
 * @Last Modified by: Songshan Gong
 * @Last Modified time: 2019-09-29 02:57:24
 */
#ifndef INCLUDE_SMARTPLUGIN_CONVERT_H_
#define INCLUDE_SMARTPLUGIN_CONVERT_H_

#include <string>
#include <vector>
#include <memory>
#include "hobotxsdk/xstream_data.h"
#include "horizon/vision_type/vision_type_util.h"
#include "smartplugin_box/traffic_info.h"
#include "xproto_msgtype/vioplugin_data.h"
#include "smartplugin_box/smartmessage.h"

namespace horizon {
namespace iot {
namespace smartplugin_multiplebox {
using horizon::vision::xproto::basic_msgtype::VioMessage;

using SnapshotInfoXStreamBaseData =
hobot::vision::SnapshotInfo<xstream::BaseDataPtr>;
using SnapshotInfoXStreamBaseDataPtr =
        std::shared_ptr<SnapshotInfoXStreamBaseData>;
using XStreamSnapshotInfo =
        xstream::XStreamData<SnapshotInfoXStreamBaseDataPtr>;
using XStreamSnapshotInfoPtr = std::shared_ptr<XStreamSnapshotInfo>;

using XStreamUserData = xstream::XStreamData<horizon::vision::xproto
    ::smartplugin_multiplebox::UserData>;
using XStreamUserDataPtr = std::shared_ptr<XStreamUserData>;
using FloatArray = horizon::vision::xproto::smartplugin_multiplebox::FloatArray;
using FeatureInfo =
        horizon::vision::xproto::smartplugin_multiplebox::FeatureInfo;
using FeatureFrameMessage =
       horizon::vision::xproto::smartplugin_multiplebox::FeatureFrameMessage;

using XStreamFeature = xstream::XStreamData<hobot::vision::Feature>;

class Convertor {
 public:
  /**
   * @brief convert input VioMessage to xstream inputdata.
   * @param input input VioMessage
   * @return xstream::InputDataPtr xstream input
   */
  static xstream::InputDataPtr ConvertInput(const VioMessage *input);

  /**
   * @brief 从xstream的输出中解析出车辆结构化信息
   */
  static std::vector<vision::xproto::smartplugin::VehicleInfo>
  ParseXstreamOutputToVehicleInfo(const xstream::OutputDataPtr xstream_outputs);

  /**
   * @brief 从xstream的输出中解析出非机动车的结构化信息
   */
  static std::vector<vision::xproto::smartplugin::NoMotorVehicleInfo>
  ParseXstreamOutputToNoMotorInfo(const xstream::OutputDataPtr xstream_outputs);

  /**
   * @brief 从xstream的输出中解析出人的结构化信息
   */
  static std::vector<vision::xproto::smartplugin::PersonInfo>
  ParseXstreamOutputToPersonInfo(const xstream::OutputDataPtr xstream_outputs);

  /**
   * @brief 从xstream的输出中解析出丢失的track
   */
  static std::vector<uint64_t> ParseXstreamOutputToLostTrack(
      const xstream::OutputDataPtr xstream_outputs);

  /**
   * @brief 模型输出的车牌需要进行转换才能输出可视化的车牌号
   *
   * @param plate_num_model 模型输出
   *
   * @return "":failed
   */
  static std::string ConvertPlateNumber(
      const std::vector<int> &plate_num_model);

  /**
   * @brief 将json转换成未格式化的字符串
   */
  //  static std::string JsonToUnStyledString(const Json::Value &jv);

  static HorizonVisionSmartFrame *ConvertOutputToSmartFrame(
      const xstream::OutputDataPtr &xroc_output);

  static void ConvertOutputToSmartFrame(
      const xstream::OutputDataPtr &xroc_output,
      HorizonVisionSmartFrame *smart_frame);

  static void ConvertOutput(const xstream::OutputDataPtr &xstream_output,
                     FeatureFrameMessage &feature_frame_message);

 public:
  static int image_compress_quality;
};

}  // namespace smartplugin_multiplebox
}  // namespace iot
}  // namespace horizon

#endif  //  INCLUDE_SMARTPLUGIN_CONVERT_H_
