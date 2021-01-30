/**
 * @copyright Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @file      bbox_filter_b.h
 * @brief     BBoxFilterB class
 * @author    Qingpeng Liu (qingpeng.liu@horizon.ai)
 * @version   0.0.0.1
 * @date      2020-01-03
 */

#ifndef XSTREAM_FRAMEWORK_TUTORIALS_STAGE7_INCLUDE_METHOD_BBOX_FILTER_B_H_
#define XSTREAM_FRAMEWORK_TUTORIALS_STAGE7_INCLUDE_METHOD_BBOX_FILTER_B_H_

#include <atomic>
#include <string>
#include <vector>
#include "hobotxstream/method.h"

namespace xstream {

class BBoxFilterB : public Method {
 public:
  int Init(const std::string &config_file_path) override;

  virtual std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<xstream::InputParamPtr> &param);

  void Finalize() override;

  int UpdateParameter(InputParamPtr ptr) override;

  InputParamPtr GetParameter() const override;

  std::string GetVersion() const override;

 private:
  std::atomic<float> area_threshold_;
};

class BBoxFilterBParam : public xstream::InputParam {
 public:
  explicit BBoxFilterBParam(const std::string &module_name) :
           xstream::InputParam(module_name) {}
  std::string Format() override {
    return "";
  }
};

}  // namespace xstream

#endif  // XSTREAM_FRAMEWORK_TUTORIALS_STAGE7_INCLUDE_METHOD_BBOX_FILTER_B_H_
