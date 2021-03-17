//
// Copyright 2019 Horizon Robotics.
//

#include "ssd_method/ssd_method.h"

#include <sys/time.h>

#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>

#include "hobotlog/hobotlog.hpp"
#include "ssd_method/ssd_predictor.h"
#include "ssd_method/ssd_result_post_process.h"

namespace xstream {

int SSDMethod::Init(const std::string &config_file_path) {
  LOGD << "SSDMethod Init " << config_file_path << std::endl;
  ssd_predictor_.reset(new xstream::SSDPredictor());
  int ret = ssd_predictor_->Init(config_file_path);
  HOBOT_CHECK(ret == 0) << "SSD_method init failed.";
  method_param_.reset(new xstream::SSDParam("SSDMethod"));
  LOGD << "Finish SSDMethod Init";
  return 0;
}

std::vector<BaseDataPtr> SSDMethod::DoProcess(
    const std::vector<BaseDataPtr> &input,
    const xstream::InputParamPtr &param) {
  LOGD << "Run SSDMethod, input's size: " << input.size();

  std::vector<BaseDataPtr> output;
  ssd_predictor_->RunSingleFrame(input, output);
  return output;
}
}  // namespace xstream
