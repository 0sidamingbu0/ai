/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: util.h
 * @Brief: declaration of the common func
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-08-26 9:47:31
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-08-26 10:22:45
 */

#ifndef COMMON_UTIL_H_
#define COMMON_UTIL_H_

#include <string>
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>
#include "opencv2/core/core.hpp"

namespace xstream {

// Align by 16
#define ALIGN_16(v) ((v + (16 - 1)) / 16 * 16)

inline void MemZero(void* p, size_t n)
{
  memset(p, 0, n);
}

inline float SigMoid(const float &input) {
  return 1 / (1 + std::exp(-1 * input));
}

inline float GetFloatByInt(int32_t value, uint32_t shift) {
  float ret_x = value;
  if (value != 0) {
    int *ix = reinterpret_cast<int *>(&ret_x);
    (*ix) -= shift * 0x00800000;
  }
  return ret_x;
}

// coordinate transform.
// fasterrcnn model's input size maybe not eqaul to origin image size,
// needs coordinate transform for detection result.
template<typename T>
inline void CoordinateTransform(
    T &x, T &y,
    int src_image_width, int src_image_height,
    int model_input_width, int model_input_hight) {
  x = x * 1.0 * src_image_width / model_input_width;
  y = y * 1.0 * src_image_height / model_input_hight;
}

// coordinate drift.
// FCOS model use crop image for original image,
// which needs coordinate drift for detection result.
template<typename T>
inline void CoordinateDrift(
    T &x, T &y,
    int x_drift, int y_drift) {
  x += x_drift;
  y += y_drift;
}

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

// 返回vector排序的索引值
template <typename T>
inline std::vector<size_t> SortIndexes(const std::vector<T> &v, bool cmp)
{
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  if (cmp) {
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
  } else {
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
  }
  return idx;
}

inline std::string GetParentPath(const std::string &path) {
  auto pos = path.rfind('/');
  if (std::string::npos != pos) {
    auto parent = path.substr(0, pos);
    return parent + "/";
  } else {
    return std::string("./");
  }
}

}  // namespace xstream

#endif  // COMMON_UTIL_H_
