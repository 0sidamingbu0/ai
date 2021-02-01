/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @brief     Average and StandardDeviation Method
 * @author    zhe.sun
 * @version   0.0.0.1
 * @date      2020.10.31
 */

#ifndef XSTREAM_FRAMEWORK_TUTORIALS_STAGE2_METHOD_METHOD_H_
#define XSTREAM_FRAMEWORK_TUTORIALS_STAGE2_METHOD_METHOD_H_

#include <string>
#include <vector>
#include "hobotxstream/method.h"
#include "json/json.h"

namespace xstream {

class Average : public Method {
 public:
  int Init(const std::string &config_file_path) override;

  std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<xstream::InputParamPtr> &param) override;

  MethodInfo GetMethodInfo() override;

  void Finalize() override;

  int UpdateParameter(InputParamPtr ptr) override;

  InputParamPtr GetParameter() const override;

  std::string GetVersion() const override;
};

class StandardDeviation : public Method {
 public:
  int Init(const std::string &config_file_path) override;

  std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<xstream::InputParamPtr> &param) override;

  MethodInfo GetMethodInfo() override;

  void Finalize() override;

  int UpdateParameter(InputParamPtr ptr) override;

  InputParamPtr GetParameter() const override;

  std::string GetVersion() const override;
};

// class ValueParam : public xstream::InputParam {
//  public:
//   explicit ValueParam(const std::string &module_name) :
//            xstream::InputParam(module_name) {}
//   std::string Format() override {
//     return "";
//   }
// };

}  // namespace xstream

#endif  // XSTREAM_FRAMEWORK_TUTORIALS_STAGE2_METHOD_METHOD_H_
