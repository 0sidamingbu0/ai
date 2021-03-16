/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include "hobotxstream/method_factory.h"
#include <string>
#include "CNNMethod/CNNMethod.h"
#include "FasterRCNNMethod/FasterRCNNMethod.h"
#include "MOTMethod/MOTMethod.h"
#include "GradingMethod/GradingMethod.h"
#include "SnapShotMethod/SnapShotMethod.h"
#include "MergeMethod/MergeMethod.h"

namespace xstream {
namespace method_factory {
MethodPtr CreateMethod(const std::string &method_name) {
  if ("FasterRCNNMethod" == method_name) {
    return MethodPtr(new FasterRCNNMethod());
  } else if ("MOTMethod" == method_name) {
    return MethodPtr(new MOTMethod());
  } else if ("CNNMethod" == method_name) {
    return MethodPtr(new CNNMethod());
  } else if ("GradingMethod" == method_name) {
    return MethodPtr(new GradingMethod());
  } else if ("SnapShotMethod" == method_name) {
    return MethodPtr(new SnapShotMethod());
  } else if ("MergeMethod" == method_name) {
    return MethodPtr(new MergeMethod());
  } else {
    return MethodPtr();
  }
}
}  // namespace method_factory
}  // namespace xstream
