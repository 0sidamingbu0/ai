/*
 * @Description: define data struct for apa
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-03 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-09-04 9:30:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef MULTISOURCE_SMARTPLUGIN_COMMON_DATA_H_
#define MULTISOURCE_SMARTPLUGIN_COMMON_DATA_H_
#include <vector>
#include <map>

#include "horizon/vision_type/vision_type.hpp"

using hobot::vision::BBox;
using hobot::vision::Segmentation;
using hobot::vision::Points;

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcedata {

enum BboxTypeEnum {
  kBboxTypeEnumCyclist = 0,  // 骑车人
  kBboxTypeEnumPerson = 1,   // 行人
  kBboxTypeEnumVehicle = 2,  // 全车
  kBboxTypeEnumVehicleRear = 3,  // 车尾
  kBboxTypeEnumParkingLock = 4,  // 地锁
  kBboxTypeEnumCornerPoint = 5,  // 角点
  kBboxTypeEnumParking = 6  // 停车位,带有旋转角度
};

enum SegmetationTypeEnum {
  kSegmetationTypeEnumFreeSpace = 0,  // 可行驶区域
  kSegmetationTypeEnumLanLineMask = 1,  //车道线分割
  kSegmetationTypeEnumParking = 2  // 停车位分割
};

enum PointsTypeEnum {
  kNone = 0    // 可用于描述直线
};

struct MultiSourceSmartResult {
  std::map<int, std::vector<BBox>> bbox_list_;
  std::map<int, std::vector<Segmentation>> segmentation_list_;
  std::map<int, std::vector<Points>> points_list_;
};
}  // namespace multisourcedata
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // MULTISOURCE_SMARTPLUGIN_COMMON_DATA_H_
