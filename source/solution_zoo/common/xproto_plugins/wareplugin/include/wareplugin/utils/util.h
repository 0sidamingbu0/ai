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

#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_msg.h"
#include "horizon/vision_type/vision_type_util.h"
#include "horizon/vision_type/vision_type.hpp"
#include "horizon/vision_type/vision_msg.hpp"
#ifndef _WAREPLUGIN_UTILS_UTIL_H_
#define _WAREPLUGIN_UTILS_UTIL_H_

namespace horizon {
namespace vision {
namespace xproto {
namespace wareplugin {

hobot::vision::Points Box2Points(const hobot::vision::BBox &box);
hobot::vision::BBox Points2Box(const hobot::vision::Points &points);

}   //  namespace wareplugin
}   //  namespace xproto
}   //  namespace vision
}   //  namespace horizon

#endif  //  _WAREPLUGIN_UTILS_UTIL_H_
