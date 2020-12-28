/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-08-12
 * @Version: v1.0.0
 * @Brief: implemenation of vio pipe cfg.
 */
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "hobotlog/hobotlog.hpp"
#include "iotviomanager/viopipeconfig.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

bool VioPipeConfig::GetConfig(void *cfg) {
  HOBOT_CHECK(cfg != nullptr) << "vio pipe config is null";
  IotVioCfg *vio_cfg = static_cast<IotVioCfg*>(cfg);

  *vio_cfg = vio_cfg_;

  return true;
}

bool VioPipeConfig::LoadConfigFile() {
  std::ifstream ifs(path_);
  if (!ifs.is_open()) {
    LOGE << "Open config file " << path_ << " failed";
    return false;
  }
  LOGI << "Open config file " << path_ << " success";
  std::stringstream ss;
  ss << ifs.rdbuf();
  ifs.close();
  std::string content = ss.str();
  Json::Value value;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  JSONCPP_STRING error;
  std::shared_ptr<Json::CharReader> reader(builder.newCharReader());
  try {
    bool ret = reader->parse(content.c_str(),
        content.c_str() + content.size(),
        &json_, &error);
    if (ret) {
      LOGE << "Open config file1 " << path_;
      HOBOT_CHECK(json_);
      return true;
    } else {
      LOGE << "Open config file2 " << path_;
      return false;
    }
  } catch (std::exception &e) {
    LOGE << "Open config file3 " << path_;
    return false;
  }
}

int VioPipeConfig::GetIntValue(const std::string &key) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (json_[key].empty()) {
    LOGE << "Can not find key: " << key;
    return -1;
  }

  return json_[key].asInt();
}

std::string VioPipeConfig::GetStringValue(const std::string &key) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (json_[key].empty()) {
    LOGE << "Can not find key: " << key;
    return "";
  }

  return json_[key].asString();
}

Json::Value VioPipeConfig::GetJson() const { return this->json_; }

bool VioPipeConfig::ParserConfig() {
  int pym_chn_num = 0;

  std::string pipe_name = "pipeline";
  std::string grp_name = "vps";
  // 1. vin vps config
  vio_cfg_.cam_en = json_[pipe_name]["cam_en"].asInt();
  vio_cfg_.vin_vps_mode = json_[pipe_name]["vin_vps_mode"].asInt();
  // 2. vin detail config
  vio_cfg_.vin_cfg.vin_fd_en = json_[pipe_name]["vin"]["vin_fd_en"].asInt();
  // 2.1 sensor config
  vio_cfg_.vin_cfg.sensor_info.sensor_id =
    json_[pipe_name]["vin"]["sensor"]["sensor_id"].asInt();
  vio_cfg_.vin_cfg.sensor_info.sensor_port =
    json_[pipe_name]["vin"]["sensor"]["sensor_port"].asInt();
  vio_cfg_.vin_cfg.sensor_info.i2c_bus =
    json_[pipe_name]["vin"]["sensor"]["i2c_bus"].asInt();
  vio_cfg_.vin_cfg.sensor_info.extra_mode =
    json_[pipe_name]["vin"]["sensor"]["extra_mode"].asInt();
  vio_cfg_.vin_cfg.sensor_info.need_clk =
    json_[pipe_name]["vin"]["sensor"]["need_clk"].asInt();
  vio_cfg_.vin_cfg.sensor_info.serdes_index =
    json_[pipe_name]["vin"]["sensor"]["serdes_index"].asInt();
  vio_cfg_.vin_cfg.sensor_info.serdes_port =
    json_[pipe_name]["vin"]["sensor"]["serdes_port"].asInt();
  if (json_[pipe_name]["vin"]["sensor"].isMember("gpio_num")) {
    vio_cfg_.vin_cfg.sensor_info.gpio_num =
      json_[pipe_name]["vin"]["sensor"]["gpio_num"].asInt();
  }
  if (json_[pipe_name]["vin"]["sensor"].isMember("gpio_pin")) {
    vio_cfg_.vin_cfg.sensor_info.gpio_pin =
      json_[pipe_name]["vin"]["sensor"]["gpio_pin"].asInt();
  }
  if (json_[pipe_name]["vin"]["sensor"].isMember("gpio_level")) {
    vio_cfg_.vin_cfg.sensor_info.gpio_level =
      json_[pipe_name]["vin"]["sensor"]["gpio_level"].asInt();
  }
  // 2.2 mipi config
  vio_cfg_.vin_cfg.mipi_info.mipi_index =
    json_[pipe_name]["vin"]["mipi"]["host_index"].asInt();
  vio_cfg_.vin_cfg.mipi_info.vc_index =
    json_[pipe_name]["vin"]["mipi"]["vc_index"].asInt();
  vio_cfg_.vin_cfg.mipi_info.dol2_vc_index =
    json_[pipe_name]["vin"]["mipi"]["dol2_vc_index"].asInt();
  // 2.3 sif config
  vio_cfg_.vin_cfg.sif_info.need_md =
    json_[pipe_name]["vin"]["sif"]["need_md"].asInt();
  vio_cfg_.vin_cfg.sif_info.sif_out_buf_num =
    json_[pipe_name]["vin"]["sif"]["sif_out_buf_num"].asInt();
  // 2.4 isp config
  vio_cfg_.vin_cfg.isp_info.temper_mode =
    json_[pipe_name]["vin"]["isp"]["temper_mode"].asInt();
  vio_cfg_.vin_cfg.isp_info.isp_out_buf_num =
    json_[pipe_name]["vin"]["isp"]["isp_out_buf_num"].asInt();
  vio_cfg_.vin_cfg.isp_info.isp_3a_en =
    json_[pipe_name]["vin"]["isp"]["isp_3a_en"].asInt();
  // 2.5 dwe config
  vio_cfg_.vin_cfg.dwe_info.ldc_en =
    json_[pipe_name]["vin"]["dwe"]["ldc"]["enable"].asInt();
  vio_cfg_.vin_cfg.dwe_info.dis_en =
    json_[pipe_name]["vin"]["dwe"]["dis"]["enable"].asInt();
  // 4. vps detail config
  vio_cfg_.vps_cfg.vps_dump_num =
    json_[pipe_name][grp_name]["vps_dump_num"].asInt();
  vio_cfg_.vps_cfg.vps_layer_dump =
    json_[pipe_name][grp_name]["vps_layer_dump"].asInt();
  vio_cfg_.vps_cfg.vps_fd_en =
    json_[pipe_name][grp_name]["vps_fd_en"].asInt();
  // 4.1 feedback config
  vio_cfg_.vps_cfg.fb_info.width =
    json_[pipe_name][grp_name]["feedback"]["fb_width"].asInt();
  vio_cfg_.vps_cfg.fb_info.height =
    json_[pipe_name][grp_name]["feedback"]["fb_height"].asInt();
  vio_cfg_.vps_cfg.fb_info.buf_num =
    json_[pipe_name][grp_name]["feedback"]["fb_buf_num"].asInt();
  vio_cfg_.vps_cfg.fb_info.inner_buf_en =
    json_[pipe_name][grp_name]["feedback"]["inner_buf_en"].asInt();
  vio_cfg_.vps_cfg.fb_info.inner_buf_type =
    json_[pipe_name][grp_name]["feedback"]["inner_buf_type"].asInt();
  vio_cfg_.vps_cfg.fb_info.bind_pipe_id =
    json_[pipe_name][grp_name]["feedback"]["bind_pipe_id"].asInt();
  // 4.2 gdc config
  for (int gdc_idx = 0; gdc_idx < MAX_GDC_NUM; gdc_idx++) {
    std::string gdc_name = "gdc" + std::to_string(gdc_idx);
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].gdc_en =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["enable"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].gdc_w =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["gdc_w"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].gdc_h =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["gdc_h"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].frame_depth =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["frame_depth"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].src_type =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["src_type"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].rotate =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["rotate"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].bind_ipu_chn =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["bind_ipu_chn"].asInt();
    vio_cfg_.vps_cfg.gdc_info[gdc_idx].file_path =
      json_[pipe_name][grp_name]["gdc"][gdc_name]["path"].asString();
  }
  // 5. chn config
  vio_cfg_.vps_chn_num = json_[pipe_name][grp_name]["ipu"]["chn_num"].asInt();
  HOBOT_CHECK(vio_cfg_.vps_chn_num <= MAX_CHN_NUM);
  for (int chn_idx = 0; chn_idx < vio_cfg_.vps_chn_num; chn_idx++) {
    std::string chn_name = "chn" + std::to_string(chn_idx);
    vio_cfg_.chn_cfg[chn_idx].ipu_chn_en =
      json_[pipe_name][grp_name]["ipu"][chn_name]["ipu_chn_en"].asInt();
    vio_cfg_.chn_cfg[chn_idx].pym_chn_en =
      json_[pipe_name][grp_name]["ipu"][chn_name]["pym_chn_en"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_frame_depth =
      json_[pipe_name][grp_name]["ipu"][chn_name]["frame_depth"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_frame_timeout =
      json_[pipe_name][grp_name]["ipu"][chn_name]["timeout"].asInt();
    // 5.1 ipu chn roi config
    vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_en =
      json_[pipe_name][grp_name]["ipu"][chn_name]["roi_en"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_x =
      json_[pipe_name][grp_name]["ipu"][chn_name]["roi_x"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_y =
      json_[pipe_name][grp_name]["ipu"][chn_name]["roi_y"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_w =
      json_[pipe_name][grp_name]["ipu"][chn_name]["roi_w"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_h =
      json_[pipe_name][grp_name]["ipu"][chn_name]["roi_h"].asInt();
    // 5.2 ipu chn scale config
    vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_w =
      json_[pipe_name][grp_name]["ipu"][chn_name]["scale_w"].asInt();
    vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_h =
      json_[pipe_name][grp_name]["ipu"][chn_name]["scale_h"].asInt();
    // 6. pym config
    if (vio_cfg_.chn_cfg[chn_idx].pym_chn_en == 1 && pym_chn_num == 0) {
      /* 6.1 pym ctrl config */
      vio_cfg_.pym_chn_index = chn_idx;
      vio_cfg_.pym_cfg.frame_id =
        json_[pipe_name][grp_name]["pym"]["pym_ctrl_config"]\
        ["frame_id"].asUInt();
      vio_cfg_.pym_cfg.ds_uv_bypass =
        json_[pipe_name][grp_name]["pym"]\
        ["pym_ctrl_config"]["ds_uv_bypass"].asUInt();
      vio_cfg_.pym_cfg.ds_layer_en =
        (uint16_t)json_[pipe_name][grp_name]["pym"]\
        ["pym_ctrl_config"]["ds_layer_en"].asUInt();
      vio_cfg_.pym_cfg.us_layer_en =
        (uint8_t)json_[pipe_name][grp_name]["pym"]\
        ["pym_ctrl_config"]["us_layer_en"].asUInt();
      vio_cfg_.pym_cfg.us_uv_bypass =
        (uint8_t)json_[pipe_name][grp_name]["pym"]\
        ["pym_ctrl_config"]["us_uv_bypass"].asUInt();
      vio_cfg_.pym_cfg.frameDepth =
        json_[pipe_name][grp_name]["pym"]\
        ["pym_ctrl_config"]["frame_depth"].asUInt();
      vio_cfg_.pym_cfg.timeout =
        json_[pipe_name][grp_name]["pym"]\
        ["pym_ctrl_config"]["timeout"].asInt();
      /* 6.2 pym downscale config */
      for (int ds_idx = 0 ; ds_idx < MAX_PYM_DS_NUM; ds_idx++) {
        if (ds_idx % 4 == 0) continue;
        std::string factor_name = "factor_" + std::to_string(ds_idx);
        std::string roi_x_name = "roi_x_" + std::to_string(ds_idx);
        std::string roi_y_name = "roi_y_" + std::to_string(ds_idx);
        std::string roi_w_name = "roi_w_" + std::to_string(ds_idx);
        std::string roi_h_name = "roi_h_" + std::to_string(ds_idx);
        vio_cfg_.pym_cfg.ds_info[ds_idx].factor =
          (uint8_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_ds_config"][factor_name].asUInt();
        vio_cfg_.pym_cfg.ds_info[ds_idx].roi_x =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_ds_config"][roi_x_name].asUInt();
        vio_cfg_.pym_cfg.ds_info[ds_idx].roi_y =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_ds_config"][roi_y_name].asUInt();
        vio_cfg_.pym_cfg.ds_info[ds_idx].roi_width =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_ds_config"][roi_w_name].asUInt();
        vio_cfg_.pym_cfg.ds_info[ds_idx].roi_height =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_ds_config"][roi_h_name].asUInt();
      }
      /* 6.3 pym upscale config */
      for (int us_idx = 0 ; us_idx < MAX_PYM_US_NUM; us_idx++) {
        std::string factor_name = "factor_" + std::to_string(us_idx);
        std::string roi_x_name = "roi_x_" + std::to_string(us_idx);
        std::string roi_y_name = "roi_y_" + std::to_string(us_idx);
        std::string roi_w_name = "roi_w_" + std::to_string(us_idx);
        std::string roi_h_name = "roi_h_" + std::to_string(us_idx);
        vio_cfg_.pym_cfg.us_info[us_idx].factor =
          (uint8_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_us_config"][factor_name].asUInt();
        vio_cfg_.pym_cfg.us_info[us_idx].roi_x =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_us_config"][roi_x_name].asUInt();
        vio_cfg_.pym_cfg.us_info[us_idx].roi_y =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_us_config"][roi_y_name].asUInt();
        vio_cfg_.pym_cfg.us_info[us_idx].roi_width =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_us_config"][roi_w_name].asUInt();
        vio_cfg_.pym_cfg.us_info[us_idx].roi_height =
          (uint16_t)json_[pipe_name][grp_name]["pym"]\
          ["pym_us_config"][roi_h_name].asUInt();
      }
      /* 6.4 calculate pym num */
      pym_chn_num++;
    }   // end pym channel config
  }  // end all channel config

  HOBOT_CHECK(pym_chn_num <= 1) << "every group only max support one pym";
  PrintConfig();
  return true;
}

bool VioPipeConfig::PrintConfig() {
  LOGI << "******** iot vio config: " << pipe_id_ << " start *********";
  // 1. vin vps config
  LOGI << "cam_en: "       << vio_cfg_.cam_en;
  LOGI << "vin_vps_mode: " << vio_cfg_.vin_vps_mode;
  // 2. vin detail config
  LOGI << "vin_fd_en: "    << vio_cfg_.vin_cfg.vin_fd_en;
  // 2.1 sensor config
  LOGI << "sensor_id: "    << vio_cfg_.vin_cfg.sensor_info.sensor_id;
  LOGI << "sensor_port: "  << vio_cfg_.vin_cfg.sensor_info.sensor_port;
  LOGI << "i2c_bus: "      << vio_cfg_.vin_cfg.sensor_info.i2c_bus;
  LOGI << "extra_mode: "      << vio_cfg_.vin_cfg.sensor_info.extra_mode;
  LOGI << "need_clk: "     << vio_cfg_.vin_cfg.sensor_info.need_clk;
  LOGI << "serdes_index: " << vio_cfg_.vin_cfg.sensor_info.serdes_index;
  LOGI << "serdes_port: "  << vio_cfg_.vin_cfg.sensor_info.serdes_port;
  LOGI << "gpio_num: "  << vio_cfg_.vin_cfg.sensor_info.gpio_num;
  LOGI << "gpio_pin: "  << vio_cfg_.vin_cfg.sensor_info.gpio_pin;
  LOGI << "gpio_level: "  << vio_cfg_.vin_cfg.sensor_info.gpio_level;
  // 2.2 mipi config
  LOGI << "mipi_index: "   << vio_cfg_.vin_cfg.mipi_info.mipi_index;
  LOGI << "vc_index: "     << vio_cfg_.vin_cfg.mipi_info.vc_index;
  LOGI << "dol2_vc_index: "  << vio_cfg_.vin_cfg.mipi_info.dol2_vc_index;
  // 2.3 sif config
  LOGI << "need_md: "      << vio_cfg_.vin_cfg.sif_info.need_md;
  LOGI << "sif_out_buf_num: "  << vio_cfg_.vin_cfg.sif_info.sif_out_buf_num;
  // 2.4 isp config
  LOGI << "temper_mode: "  << vio_cfg_.vin_cfg.isp_info.temper_mode;
  LOGI << "isp_out_buf_num: "  << vio_cfg_.vin_cfg.isp_info.isp_out_buf_num;
  LOGI << "isp_3a_en: "  << vio_cfg_.vin_cfg.isp_info.isp_3a_en;
  // 2.5 dwe config
  LOGI << "ldc_en: "       << vio_cfg_.vin_cfg.dwe_info.ldc_en;
  LOGI << "dis_en: "       << vio_cfg_.vin_cfg.dwe_info.dis_en;
  // 3. vps detail config
  LOGI << "vps_dump_num: "        << vio_cfg_.vps_cfg.vps_dump_num;
  LOGI << "vps_layer_dump: "  << vio_cfg_.vps_cfg.vps_layer_dump;
  LOGI << "vps_fd_en: "       << vio_cfg_.vps_cfg.vps_fd_en;
  // 3.1 feedback config
  LOGI << "fb_width: "     << vio_cfg_.vps_cfg.fb_info.width;
  LOGI << "fb_height: "    << vio_cfg_.vps_cfg.fb_info.height;
  LOGI << "fb_buf_num: "   << vio_cfg_.vps_cfg.fb_info.buf_num;
  LOGI << "fb_inner_buf_en: "   << vio_cfg_.vps_cfg.fb_info.inner_buf_en;
  LOGI << "fb_inner_buf_type: "   << vio_cfg_.vps_cfg.fb_info.inner_buf_type;
  LOGI << "fb_bind_pipe_id: "   << vio_cfg_.vps_cfg.fb_info.bind_pipe_id;
  // 3.2 gdc config
  for (int gdc_idx = 0; gdc_idx < MAX_GDC_NUM; gdc_idx++) {
    LOGI << "gdc_" << gdc_idx << "_en: "
      << vio_cfg_.vps_cfg.gdc_info[gdc_idx].gdc_en;
    LOGI << "gdc_" << gdc_idx << "_frame_depth: "
      << vio_cfg_.vps_cfg.gdc_info[gdc_idx].frame_depth;
    LOGI << "gdc_" << gdc_idx << "_src_type: "
      << vio_cfg_.vps_cfg.gdc_info[gdc_idx].src_type;
    LOGI << "gdc_" << gdc_idx << "_rotate: "
      << vio_cfg_.vps_cfg.gdc_info[gdc_idx].rotate;
    LOGI << "gdc_" << gdc_idx << "_bid_ipu_chn: "
      << vio_cfg_.vps_cfg.gdc_info[gdc_idx].bind_ipu_chn;
    LOGI << "gdc_" << gdc_idx << "_file_path: "
      << vio_cfg_.vps_cfg.gdc_info[gdc_idx].file_path;
  }
  // 4. chn config
  LOGI << "vps_chn_num: "       << vio_cfg_.vps_chn_num;
  for (int chn_idx = 0 ; chn_idx < vio_cfg_.vps_chn_num; chn_idx++) {
    LOGI << "chn_index: "<< chn_idx << " ipu_chn_en: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_chn_en;
    LOGI << "chn_index: "<< chn_idx << " pym_chn_en: "
      << vio_cfg_.chn_cfg[chn_idx].pym_chn_en;
    LOGI << "chn_index: "<< chn_idx << " ipu_frame_depth: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_frame_depth;
    LOGI << "chn_index: "<< chn_idx << " ipu_frame_timeout: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_frame_timeout;
    // 4.1 ipu chn scale config
    LOGI << "chn_index: "<< chn_idx << " scale_w: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_w;
    LOGI << "chn_index: "<< chn_idx << " scale_h: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_scale.scale_h;
    // 4.2 ipu chn roi config
    LOGI << "chn_index: "<< chn_idx << " roi_en: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_en;
    LOGI << "chn_index: "<< chn_idx << " roi_x: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_x;
    LOGI << "chn_index: "<< chn_idx << " roi_y: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_y;
    LOGI << "chn_index: "<< chn_idx << " roi_w: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_w;
    LOGI << "chn_index: "<< chn_idx << " roi_h: "
      << vio_cfg_.chn_cfg[chn_idx].ipu_roi.roi_h;
    // 5. pym config
    if (vio_cfg_.chn_cfg[chn_idx].pym_chn_en == 1) {
      LOGI << "--------chn:" << chn_idx << " pym config start---------";
      LOGI << "chn_index: "<< chn_idx << " " << "frame_id: "
        << vio_cfg_.pym_cfg.frame_id;
      LOGI << "chn_index: "<< chn_idx << " " << "ds_layer_en: "
        << vio_cfg_.pym_cfg.ds_layer_en;
      LOGI << "chn_index: "<< chn_idx << " " << "ds_uv_bypass: "
        << vio_cfg_.pym_cfg.ds_uv_bypass;
      LOGI << "chn_index: "<< chn_idx << " " << "us_layer_en: "
        << static_cast<int>(vio_cfg_.pym_cfg.us_layer_en);
      LOGI << "chn_index: "<< chn_idx << " " << "us_uv_bypass: "
        << static_cast<int>(vio_cfg_.pym_cfg.us_uv_bypass);
      LOGI << "chn_index: "<< chn_idx << " " << "frameDepth: "
        << vio_cfg_.pym_cfg.frameDepth;
      LOGI << "chn_index: "<< chn_idx << " " << "timeout: "
        << vio_cfg_.pym_cfg.timeout;
      // 5.2 pym downscale config
      for (int ds_idx = 0 ; ds_idx < MAX_PYM_DS_NUM; ds_idx++) {
        if (ds_idx % 4 == 0) continue;
        LOGI << "ds_pym_layer: "<< ds_idx << " " << "ds roi_x: "
          << vio_cfg_.pym_cfg.ds_info[ds_idx].roi_x;
        LOGI << "ds_pym_layer: "<< ds_idx << " " << "ds roi_y: "
          << vio_cfg_.pym_cfg.ds_info[ds_idx].roi_y;
        LOGI << "ds_pym_layer: "<< ds_idx << " " << "ds roi_width: "
          << vio_cfg_.pym_cfg.ds_info[ds_idx].roi_width;
        LOGI << "ds_pym_layer: "<< ds_idx << " " << "ds roi_height: "
          << vio_cfg_.pym_cfg.ds_info[ds_idx].roi_height;
        LOGI << "ds_pym_layer: "<< ds_idx << " " << "ds factor: "
          << static_cast<int>(vio_cfg_.pym_cfg.ds_info[ds_idx].factor);
      }
      /* 4.3 pym upscale config */
      for (int us_idx = 0 ; us_idx < MAX_PYM_US_NUM; us_idx++) {
        LOGI << "us_pym_layer: "<< us_idx << " " << "us roi_x: "
          << vio_cfg_.pym_cfg.us_info[us_idx].roi_x;
        LOGI << "us_pym_layer: "<< us_idx << " " << "us roi_y: "
          << vio_cfg_.pym_cfg.us_info[us_idx].roi_y;
        LOGI << "us_pym_layer: "<< us_idx << " " << "us roi_width: "
          << vio_cfg_.pym_cfg.us_info[us_idx].roi_width;
        LOGI << "us_pym_layer: "<< us_idx << " " << "us roi_height: "
          << vio_cfg_.pym_cfg.us_info[us_idx].roi_height;
        LOGI << "us_pym_layer: "<< us_idx << " " << "us factor: "
          << static_cast<int>(vio_cfg_.pym_cfg.us_info[us_idx].factor);
      }
      LOGI << "---------pym_chn:" << chn_idx << " config end----------";
    }
  }
  LOGI << "******** iot vio config: " << pipe_id_ << " end *********";

  return true;
}

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
