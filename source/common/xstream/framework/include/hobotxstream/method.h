/**
 * Copyright (c) 2018 Horizon Robotics. All rights reserved.
 * @brief     Method interface of xstream framework
 * @file method.h
 * @author    shuhuan.sun
 * @email     shuhuan.sun@horizon.ai
 * @version   0.0.0.1
 * @date      2018.11.21
 */
#ifndef HOBOTXSTREAM_METHOD_H_
#define HOBOTXSTREAM_METHOD_H_

#include <memory>
#include <string>
#include <vector>

#include "hobotxsdk/xstream_data.h"

namespace xstream {
// Method info
struct MethodInfo {
  // is thread safe
  bool is_thread_safe_ = false;
  // is need reorder, the order of outputdata must be same as the inputdata
  bool is_need_reorder = false;
  // is dependent on inputdata source
  bool is_src_ctx_dept = false;
};

class Method {
 public:
  virtual ~Method();
  // Init
  virtual int Init(const std::string &config_file_path) = 0;
  // Process function
  // <parameter> input: input data, input[i][j]: batch i, slot j
  virtual std::vector<std::vector<BaseDataPtr>> DoProcess(
      const std::vector<std::vector<BaseDataPtr>> &input,
      const std::vector<InputParamPtr> &param) = 0;
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
};

typedef std::shared_ptr<Method> MethodPtr;

}  // namespace xstream

#endif  // HOBOTXSTREAM_METHOD_H_
