/**
 * @copyright Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file      method_factory.cc
 * @brief     MethodFactory class implementation
 * @author    zhe.sun
 * @version   0.0.0.1
 * @date      2020/12/24
 */

#include "hobotxstream/method_factory.h"
#include "Yolov3PredictMethod/yolov3_predict_method.h"
#include "Yolov3PostProcessMethod/yolov3_post_process_method.h"
#include "Mobilenetv2PredictMethod/mobilenetv2_predict_method.h"
#include "Mobilenetv2PostProcessMethod/mobilenetv2_post_process_method.h"

namespace xstream {
namespace method_factory {

MethodPtr CreateMethod(const std::string &method_name) {
  if ("Yolov3PredictMethod" == method_name) {
    return MethodPtr(new Yolov3PredictMethod());
  } else if ("Yolov3PostProcessMethod" == method_name) {
    return MethodPtr(new Yolov3PostProcessMethod());
  } else if ("Mobilenetv2PredictMethod" == method_name) {
    return MethodPtr(new Mobilenetv2PredictMethod());
  } else if ("Mobilenetv2PostProcessMethod" == method_name) {
    return MethodPtr(new Mobilenetv2PostProcessMethod());
  } else {
    return MethodPtr();
  }
}

}  // namespace method_factory
}  // namespace xstream
