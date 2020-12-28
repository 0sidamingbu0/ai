/**
 * @file method_factory.cpp
 * @author hangjun.yang
 * @brief 
 * @version 0.1
 * @date 2020-12-25
 *
 * @copyright Copyright (c) 2018
 *
 */

#include "hobotxstream/method_factory.h"
#include <string>
#include "ExampleSmartPlugin.h"
#include "ExampleWebsocketPlugin.h"
#include "mobilenetv2_post_process_method.h"
#include "mobilenetv2_predict_method.h"
#include "yolov3_post_process_method.h"
#include "yolov3_predict_method.h"

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
}  //  namespace method_factory
}  //  namespace xstream
