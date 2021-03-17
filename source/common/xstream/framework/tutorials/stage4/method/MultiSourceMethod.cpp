/**
 * Copyright (c) 2019 Horizon Robotics. All rights reserved.
 * @file MultiSourceMethod.cpp
 * @brief
 * @author xudong.du
 * @email xudong.du@horizon.ai
 * @date 2020/10/29
 */
#include "MultiSourceMethod.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#if defined(HR_POSIX)
#include <pthread.h>
#endif
#include "hobotxsdk/xstream_data.h"
#include "json/json.h"
#include "hobotxstream/profiler.h"
#include "hobotlog/hobotlog.hpp"

namespace xstream {
// C++ 11 要求静态成员变量在使用之前必须声明。
std::atomic_ulong MultiSourceMethod::instance_count_;
MethodInfo MultiSourceMethod::methodinfo_;

std::vector<BaseDataPtr> MultiSourceMethod::DoProcess(
    const std::vector<BaseDataPtr> &input,
    const InputParamPtr &param) {
  std::vector<BaseDataPtr> output;
  // input data can be a batch of frame.
  // first layer of vector is the frame batch
  output.resize(input.size());
  for (size_t slot_idx = 0; slot_idx < input.size(); slot_idx++) {
    if (input[slot_idx]->state_ == DataState::INVALID) {
      LOGE << "input slot " << slot_idx << " is invalid";
      continue;
    }
    auto in_slot = std::static_pointer_cast<MultiSourceInput>(input[slot_idx]);
    auto out_slot = std::make_shared<MultiSourceOutput>();

    sum_ += in_slot->sum_in;
    out_slot->sum_out = sum_;
    out_slot->method_id = method_id_;
    output[slot_idx] = out_slot;
  }
  return output;
}

void MultiSourceMethod::SetMethodInfo(const MethodInfo &methodinfo) {
  methodinfo_ = methodinfo;
}
}  // namespace xstream
