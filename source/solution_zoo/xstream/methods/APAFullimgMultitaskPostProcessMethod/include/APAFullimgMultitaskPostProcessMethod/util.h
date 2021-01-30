/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: util.h
 * @Brief: declaration of the util
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2021-01-09 9:47:31
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2021-01-09 10:22:45
 */

#ifndef INCLUDE_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_UTIL_H_
#define INCLUDE_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_UTIL_H_

#include <string>
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>
#include "opencv2/core/core.hpp"

namespace xstream {

// numpy arange
template<typename T>
inline std::vector<T> Arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

// numpy meshgrid
inline void MeshGrid(const cv::Mat &xgv, const cv::Mat &ygv,
              cv::Mat &X, cv::Mat &Y)
{
  cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
  cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

}  // namespace xstream

#endif  // INCLUDE_APAFULLIMGMULTITASKPOSTPROCESSMETHOD_UTIL_H_
