/*
 * @Description: implement of data_type
 * @Author: ronghui.zhang@horizon.ai
 * @Date: 2020-11-02
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MERGEMETHOD_DATA_TYPE_DATA_TYPE_H_
#define INCLUDE_MERGEMETHOD_DATA_TYPE_DATA_TYPE_H_

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hobotlog/hobotlog.hpp"
#include "hobotxstream/method.h"
#include "horizon/vision_type/vision_error.h"
#include "horizon/vision_type/vision_type.hpp"
#include "json/json.h"

namespace xstream {

using hobot::vision::BBox;
using hobot::vision::Landmarks;

using xstream::BaseData;
using xstream::BaseDataPtr;
using xstream::BaseDataVector;

typedef xstream::XStreamData<float> XStreamFloat;
typedef xstream::XStreamData<uint32_t> XStreamUint32;

typedef xstream::XStreamData<BBox> XStreamBBox;
typedef std::shared_ptr<XStreamBBox> XStreamBBoxPtr;

typedef xstream::XStreamData<Landmarks> XStreamKps;
typedef xstream::XStreamData<Landmarks> XStreamLandmarks;
typedef std::shared_ptr<BaseDataVector> BaseDataVectorPtr;

#define SET_SNAPSHOT_METHOD_PARAM(json_cfg, type, key)      \
  if (json_cfg.isMember(#key) && json_cfg[#key].is##type()) \
  key = json_cfg[#key].as##type()

inline float MoveCoord(const float &origin_coord, const float &offset_ratio,
                       const float &offset_length) {
  return (origin_coord + offset_ratio * offset_length);
}

inline void DumpBox(const std::shared_ptr<XStreamBBox> &box) {
  std::cout << box->value.x1 << std::endl;
  std::cout << box->value.y1 << std::endl;
  std::cout << box->value.x2 << std::endl;
  std::cout << box->value.y2 << std::endl;
}

inline float intersection(const BBox &box1, const BBox &box2) {
  auto x1 = (std::max)(box1.x1, box2.x1);
  auto y1 = (std::max)(box1.y1, box2.y1);
  auto x2 = (std::min)(box1.x2, box2.x2);
  auto y2 = (std::min)(box1.y2, box2.y2);
  if (x1 >= x2 || y1 >= y2) {
    return 0;
  }
  return (x2 - x1 + 1) * (y2 - y1 + 1);
}

enum class Type { Face, Head, body };

typedef struct BBoxOffset_ {
  float x_offset_ratio;
  float y_offset_ratio;
} BBoxOffset;

class JsonReader {
 public:
  explicit JsonReader(std::string path) : path_(path) {}
  explicit JsonReader(Json::Value root) : root_(root) {}

  int32_t ParseJsonFile() {
    std::ifstream ifs(path_);
    if (!ifs.is_open()) {
      LOGE << "Open config file " << path_ << " fail!!!";
      return -1;
    }
    LOGD << "Path: " << path_;
    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();
    std::string content = ss.str();
    LOGD << "Load content: " << content;
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    JSONCPP_STRING error;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    try {
      bool ret = reader->parse(
          content.c_str(), content.c_str() + content.size(), &root_, &error);
      if (ret) {
        return 0;
      } else {
        return -1;
      }
    } catch (std::exception &e) {
      return -1;
    }
  }

  int GetIntValue(const std::string &key, int default_value = 0) {
    LOGD << "Key value: " << key;
    auto value_js = root_[key];
    if (value_js.isNull()) {
      LOGE << "Get default value";
      return default_value;
    }
    return value_js.asInt();
  }

  bool GetBoolValue(const std::string &key, bool default_value = false) {
    auto value_js = root_[key];
    if (value_js.isNull()) {
      LOGE << "Get default value";
      return default_value;
    }
    return value_js.asBool();
  }

  float GetFloatValue(const std::string &key, float default_value = 0.0) {
    auto value_js = root_[key];
    if (value_js.isNull()) {
      LOGE << "Get default value";
      return default_value;
    }
    return value_js.asFloat();
  }

  std::string GetStringValue(const std::string &key,
                             std::string default_value = "") {
    auto value_js = root_[key];
    if (value_js.isNull()) {
      LOGE << "Get default value";
      return default_value;
    }
    return value_js.asString();
  }

  std::vector<int32_t> GetIntArray(const std::string &key) {
    auto value_js = root_[key];
    std::vector<int32_t> ret;
    if (value_js.isNull()) {
      LOGE << "Get default value";
      return ret;
    }
    ret.resize(value_js.size());
    for (Json::ArrayIndex i = 0; i < value_js.size(); ++i) {
      ret[i] = value_js[i].asInt();
    }
    return ret;
  }

  std::vector<std::string> GetStringArray(const std::string &key) {
    auto value_js = root_[key];
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

  std::shared_ptr<JsonReader> GetSubConfig(const std::string &key) {
    auto value_js = root_[key];
    if (value_js.isNull()) {
      return nullptr;
    }
    return std::make_shared<JsonReader>(value_js);
  }

  template <typename T>
  void SetValue(const std::string &key, const T &value) {
    root_[key] = value;
  }

  bool HasKey(const std::string &key) {
    auto value_js = root_[key.c_str()];
    return !value_js.isNull();
  }
  std::string GetJson() { return root_.toStyledString(); }

  Json::Value GetRawJson() { return root_; }

 private:
  std::string path_;
  Json::Value root_;
};

typedef std::shared_ptr<JsonReader> JsonReaderPtr;

struct LowPassFilterParam : public InputParam {
 public:
  LowPassFilterParam() : InputParam("LowPassFilterMethod") {
    is_enable_this_method_ = true;
    is_json_format_ = true;
  }

  virtual int UpdateParameter(const JsonReaderPtr &reader);

  float alpha = 0.4;
  // 0: default, FILTERED; 1: INVALID
//  int filtered_box_state_type = 0;
  std::shared_ptr<JsonReader> reader_;
  std::string Format() override;
};

}  // namespace xstream

#endif  // INCLUDE_MERGEMETHOD_DATA_TYPE_DATA_TYPE_H_
