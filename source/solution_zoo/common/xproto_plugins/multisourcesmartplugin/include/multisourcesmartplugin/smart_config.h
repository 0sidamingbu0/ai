/*
 * @Description: implement of multi smart plugin
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-08-26 09:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-29 22:45:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCESMARTPLUGIN_SMART_CONFIG_H_
#define INCLUDE_MULTISOURCESMARTPLUGIN_SMART_CONFIG_H_
#include <string.h>
#include <memory>
#include <string>
#include <vector>
#include "json/json.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcesmartplugin {

class JsonConfigWrapper {
 public:
  explicit JsonConfigWrapper(Json::Value config) : config_(config) {}

  int GetIntValue(std::string key, int default_value = 0) {
    auto value_js = config_[key.c_str()];
    if (value_js.isNull()) {
      return default_value;
    }
    return value_js.asInt();
  }

  bool GetBoolValue(std::string key, bool default_value = false) {
    auto value_int = GetIntValue(key, default_value);
    return value_int == 0 ? false : true;
  }

  float GetFloatValue(std::string key, float default_value = 0.0) {
    auto value_js = config_[key.c_str()];
    if (value_js.isNull()) {
      return default_value;
    }
    return value_js.asFloat();
  }

  std::string GetSTDStringValue(std::string key,
                                std::string default_value = "") {
    auto value_js = config_[key.c_str()];
    if (value_js.isNull()) {
      return default_value;
    }
    return value_js.asString();
  }

  std::vector<std::string> GetSTDStringArray(std::string key) {
    auto value_js = config_[key.c_str()];
    std::vector<std::string> ret;
    if (value_js.isNull()) {
      return ret;
    }
    ret.resize(value_js.size());
    for (Json::ArrayIndex i = 0; i < value_js.size(); ++i) {
      ret[i] = value_js[i].asString();
    }
    return ret;
  }

  std::vector<int> GetIntArray(std::string key) {
    auto value_js = config_[key.c_str()];
    std::vector<int> ret;
    if (value_js.isNull()) {
      return ret;
    }
    ret.resize(value_js.size());
    for (Json::ArrayIndex i = 0; i < value_js.size(); ++i) {
      ret[i] = value_js[i].asInt();
    }
    return ret;
  }

  std::shared_ptr<JsonConfigWrapper> GetSubConfig(std::string key) {
    auto value_js = config_[key.c_str()];
    if (value_js.isNull()) {
      return nullptr;
    }
    return std::shared_ptr<JsonConfigWrapper>(new JsonConfigWrapper(value_js));
  }

  std::shared_ptr<JsonConfigWrapper> GetSubConfig(int key) {
    auto value_js = config_[key];
    if (value_js.isNull()) {
      return nullptr;
    }
    return std::shared_ptr<JsonConfigWrapper>(new JsonConfigWrapper(value_js));
  }

  bool HasMember(std::string key) { return config_.isMember(key); }
  int ItemCount(void) { return config_.size(); }

 protected:
  Json::Value config_;
};
}  // namespace multisourcesmartplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_MULTISOURCESMARTPLUGIN_SMART_CONFIG_H_
