/**
 * Copyright (c) 2018 Horizon Robotics. All rights reserved.
 * @brief     Method interface of xstream framework
 * @file      simple_method.h
 * @author    shuhuan.sun
 * @email     shuhuan.sun@horizon.ai
 * @version   0.0.0.1
 * @date      2018.11.21
 */
#ifndef HOBOTXSTREAM_SIMPLE_METHOD_H_
#define HOBOTXSTREAM_SIMPLE_METHOD_H_

#include <memory>
#include <string>
#include <vector>

#include "hobotxsdk/xstream_data.h"
#include "hobotxstream/method.h"

namespace xstream {
class SimpleMethod : public Method{
 public:
  virtual ~SimpleMethod();
  // Init
  virtual int Init(const std::string &config_file_path) = 0;
  // Process function
  // <parameter> input: input data, input[i]: slot i
  virtual std::vector<BaseDataPtr> DoProcess(
      const std::vector<BaseDataPtr> &input, const InputParamPtr &param) = 0;
  // Finalize
  virtual void Finalize() = 0;

  // Init from json string
  virtual int InitFromJsonString(const std::string &config);
  // Update method parameter
  virtual int UpdateParameter(InputParamPtr ptr);
  // Get method parameter
  virtual InputParamPtr GetParameter() const;
  // Get method version
  virtual std::string GetVersion() const;
  // Get method info
  virtual MethodInfo GetMethodInfo();
  // <parameter> input: input data, input[i][j]: batch i, slot j
  std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<InputParamPtr> &param) final;
};

typedef std::shared_ptr<SimpleMethod> SimpleMethodPtr;

}  // namespace xstream

#endif  // HOBOTXSTREAM_SIMPLE_METHOD_H_
