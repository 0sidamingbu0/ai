/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail:
 * @Date:
 * @Version:
 * @Brief:
 * @Last Modified by:
 * @Last Modified time:
 */

#ifndef _SMARTPLUGIN_INCLUDE_SMARTPLUGIN_METHOD_CONFIGER_H_
#define _SMARTPLUGIN_INCLUDE_SMARTPLUGIN_METHOD_CONFIGER_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include "smartplugin/utils/method_const.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
#include "horizon/vision_type/vision_error.h"
#include "json/json.h"
#include "hobotlog/hobotlog.hpp"
#include "hobotxsdk/xstream_data.h"
#include "hobotxsdk/xstream_sdk.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace smartplugin {

using JsonParam = xstream::SdkCommParam;

struct JsonWriter {
  std::string GetJson() { return jsonRoot.toStyledString(); }
  bool r_updated_ = false;
  bool w_updated_ = false;

  template <typename T>
  void SetValue(const std::string &key, const T &value) {
    if (jsonRoot.isMember(key)) {
      auto origin_value = jsonRoot[key];
      w_updated_ = origin_value != value || w_updated_;
    } else {
      w_updated_ = true;
    }
    jsonRoot[key] = value;
  }

  Json::Value jsonRoot;
};

struct MethodConfigInfo {
  MethodConfigInfo() {
    json_writer_ = std::make_shared<JsonWriter>();
  }
  ~MethodConfigInfo() {
    json_writer_ = nullptr;
  }
  std::string work_mode_ = kMethodWork;
  std::shared_ptr<JsonWriter> json_writer_;
};


struct MethodConfiger{
  static std::shared_ptr<MethodConfiger> Get() {
    static auto inst = std::make_shared<MethodConfiger>();
    return inst;
  }
  MethodConfiger() = default;
  ~MethodConfiger() { method_info_.clear(); }
  int HandleAPConfig(x3::Config &config);  //NOLINT
  void BuildInputParam(xstream::InputDataPtr inputdata);

  std::weak_ptr<xstream::XStreamSDK> weak_sdk_;

 private:
  std::unordered_map<std::string, MethodConfigInfo> method_info_;
  int ParseMethodConfig(xstream::InputParamPtr param_ptr,
      Json::Value *config_json);
  void SetMethodConfigDirty();
  int SetMethodConfigToXstream();

  std::vector<xstream::InputParamPtr>
  BuildMethodInputParam(const std::string &method_name) const;
  bool NeedToRunMethod(const std::string &method_name) const;

  int SetMethodthreshold(const std::string &method_name,
      const std::string &key, float value);
  void DoSetMethodthreshold(const std::string &method_name,
      const std::string &key, float value);
  int GetMethodthreshold(const std::string &module_name,
      const std::string &key, float &value);  //NOLINT

  int DoSetMethodMode(const std::string &method_name, const std::string &mode);
  int SetMethodmode(const std::string &method_name,
      const std::string &key, const std::string &mode);
  int GetMethodmode(const std::string &method_name,
      const std::string &key, std::string &value);  //NOLINT

  int GetMethodNameandRealKey(std::string *key, std::string *module_name);

  void DumpMethodInto();
};

class PassThroughDisableParam : public xstream::DisableParam {
 public:
  explicit PassThroughDisableParam(const std::string &method_name)
      : xstream::DisableParam(method_name) {
    mode_ = Mode::PassThrough;
    is_json_format_ = false;
  }
  std::string Format() override { return "pass-through"; }
};

class CropOnlyDisableParam : public JsonParam {
 public:
  explicit CropOnlyDisableParam(const std::string &method_name)
      : JsonParam(method_name,
                  "{\n"
                  "  \"snapshot_type\": \"crop\"\n"
                  "}") {}
};

}  // namespace smartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  //  _SMARTPLUGIN_INCLUDE_SMARTPLUGIN_METHOD_CONFIGER_H_
