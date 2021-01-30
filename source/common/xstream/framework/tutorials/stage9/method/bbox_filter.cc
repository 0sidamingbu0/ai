/**
 * Copyright (c) 2018 Horizon Robotics. All rights reserved.
 * @brief     BBoxFilter Method
 * @author    shuhuan.sun
 * @email     shuhuan.sun@horizon.ai
 * @version   0.0.0.1
 * @date      2018.11.23
 */

#include "method/bbox_filter.h"
#include <iostream>
#include <fstream>
#include <memory>
#include "method/bbox.h"
#include "hobotxsdk/xstream_data.h"
#include "json/json.h"
#include "hobotlog/hobotlog.hpp"

namespace xstream {

int BBoxFilter::Init(const std::string &config_file_path) {
  std::cout << "BBoxFilter::Init " << config_file_path << std::endl;
  return 0;
}

int BBoxFilter::UpdateParameter(InputParamPtr ptr) {
  return 0;
}

InputParamPtr BBoxFilter::GetParameter() const {
  auto param = std::make_shared<BBoxFilterParam>("");
  return param;
}

void BBoxFilter::Finalize() {
  std::cout << "BBoxFilter::Finalize" << std::endl;
}

std::string BBoxFilter::GetVersion() const {
  return "";
}

std::vector<std::vector<BaseDataPtr>> BBoxScoreFilter::DoProcess(
    const std::vector<std::vector<BaseDataPtr>> &input,
    const std::vector<InputParamPtr> &param) {
  std::cout << "BBoxScoreFilter::DoProcess " << input.size() << std::endl;
  std::vector<std::vector<BaseDataPtr>> output;
  output.resize(input.size());  // batch size
  // one batch
  for (size_t i = 0; i < input.size(); ++i) {
    auto &in_batch_i = input[i];
    auto &out_batch_i = output[i];
    // one slot
    for (size_t j = 0; j < in_batch_i.size(); j++) {
      out_batch_i.push_back(std::make_shared<BaseDataVector>());
      if (in_batch_i[j]->state_ == DataState::INVALID) {
        std::cout << "input slot " << j << " is invalid" << std::endl;
        continue;
      }
      auto in_rects = std::static_pointer_cast<BaseDataVector>(in_batch_i[j]);
      auto out_rects = std::static_pointer_cast<BaseDataVector>(out_batch_i[j]);
      for (auto &in_rect : in_rects->datas_) {
        auto bbox = std::static_pointer_cast<BBox>(in_rect);
        if (bbox->score > score_threshold_) {
          out_rects->datas_.push_back(in_rect);
        } else {
          std::cout << "filter out: "
                    << bbox->x1 << ","
                    << bbox->y1 << ","
                    << bbox->x2 << ","
                    << bbox->y2 << ", score: "
                    << bbox->score << std::endl;
        }
      }
    }
  }
  return output;
}

std::vector<std::vector<BaseDataPtr>> BBoxLengthFilter::DoProcess(
    const std::vector<std::vector<BaseDataPtr>> &input,
    const std::vector<InputParamPtr> &param) {
  std::cout << "BBoxScoreFilter::DoProcess " << input.size() << std::endl;
  std::vector<std::vector<BaseDataPtr>> output;
  output.resize(input.size());  // batch size
  // one batch
  for (size_t i = 0; i < input.size(); ++i) {
    auto &in_batch_i = input[i];
    auto &out_batch_i = output[i];
    // one slot
    for (size_t j = 0; j < in_batch_i.size(); j++) {
      out_batch_i.push_back(std::make_shared<BaseDataVector>());
      if (in_batch_i[j]->state_ == DataState::INVALID) {
        std::cout << "input slot " << j << " is invalid" << std::endl;
        continue;
      }
      auto in_rects = std::static_pointer_cast<BaseDataVector>(in_batch_i[j]);
      auto out_rects = std::static_pointer_cast<BaseDataVector>(out_batch_i[j]);
      for (auto &in_rect : in_rects->datas_) {
        auto bbox = std::static_pointer_cast<BBox>(in_rect);
        if (bbox->Width() > length_threshold_ &&
            bbox->Height() > length_threshold_) {
          out_rects->datas_.push_back(in_rect);
        } else {
          std::cout << "filter out: "
                    << bbox->x1 << ","
                    << bbox->y1 << ","
                    << bbox->x2 << ","
                    << bbox->y2 << ", width: "
                    << bbox->Width() << ", height: "
                    << bbox->Height() << std::endl;
        }
      }
    }
  }
  return output;
}

std::vector<std::vector<BaseDataPtr>> BBoxAreaFilter::DoProcess(
    const std::vector<std::vector<BaseDataPtr>> &input,
    const std::vector<InputParamPtr> &param) {
  std::cout << "BBoxScoreFilter::DoProcess " << input.size() << std::endl;
  std::vector<std::vector<BaseDataPtr>> output;
  output.resize(input.size());  // batch size
  // one batch
  for (size_t i = 0; i < input.size(); ++i) {
    auto &in_batch_i = input[i];
    auto &out_batch_i = output[i];
    // one slot
    for (size_t j = 0; j < in_batch_i.size(); j++) {
      out_batch_i.push_back(std::make_shared<BaseDataVector>());
      if (in_batch_i[j]->state_ == DataState::INVALID) {
        std::cout << "input slot " << j << " is invalid" << std::endl;
        continue;
      }
      auto in_rects = std::static_pointer_cast<BaseDataVector>(in_batch_i[j]);
      auto out_rects = std::static_pointer_cast<BaseDataVector>(out_batch_i[j]);
      for (auto &in_rect : in_rects->datas_) {
        auto bbox = std::static_pointer_cast<BBox>(in_rect);
        if (bbox->Width() * bbox->Height() > area_threshold_) {
          out_rects->datas_.push_back(in_rect);
        } else {
          std::cout << "filter out: "
                    << bbox->x1 << ","
                    << bbox->y1 << ","
                    << bbox->x2 << ","
                    << bbox->y2 << ", area: "
                    << bbox->Width() * bbox->Height() << std::endl;
        }
      }
    }
  }
  return output;
}

}  // namespace xstream
