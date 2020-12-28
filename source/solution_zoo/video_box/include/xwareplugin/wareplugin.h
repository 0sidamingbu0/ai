/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail:
 * @Date: 2020-11-02 20:38:52
 * @Version: v0.0.1
 * @Last Modified by:
 * @Last Modified time: 2020-11-02 20:38:52
 */

#include <string>
#include <memory>
#include <utility>
#include "xwareplugin/jsonConfigWrapper.h"
#include "xwareplugin/waredb.h"
#ifndef INCLUDE_WAREPLUGIN_WAREPLUGIN_H_
#define INCLUDE_WAREPLUGIN_WAREPLUGIN_H_
namespace horizon {
namespace vision {
namespace xproto {
namespace wareplugin {

class WarePlugin : public XPluginAsync {
 public:
  WarePlugin() = default;
  explicit WarePlugin(const std::string& config_file);

  ~WarePlugin() = default;
  int Init() override;
  int Start() override;
  int Stop() override;
  int DeInit() override;
  std::string desc() const { return "WarePlugin"; }
  int db_list();
  int db_table_clear();

 private:
  int OnGetFeatureResult(const XProtoMessagePtr& msg);
  void ParseConfig();

  std::string config_file_;
  std::shared_ptr<JsonConfigWrapper> config_;
  std::string db_name_ = "aiexpress";
  std::string db_modul_version_ = "X2_1.6";
  std::shared_ptr<DB::DB_TABLE> db_table_;

 private:
  std::string db_path_ = "./";
  bool is_recognize_ = false;
  bool is_add_record_ = false;
  float similar_thres_ = 0.74;
  bool isinit_ = false;
  unsigned int g_recod_id_;
  float db_dis_thr_ = 0.2;
  float db_sim_thr_ = 0.95;
  float recog_dis_thr_ = 1.1;
  float recog_sim_thr_ = 0.75;
};

}   //  namespace wareplugin
}   //  namespace xproto
}   //  namespace vision
}   //  namespace horizon

#endif  //  INCLUDE_WAREPLUGIN_WAREPLUGIN_H_
