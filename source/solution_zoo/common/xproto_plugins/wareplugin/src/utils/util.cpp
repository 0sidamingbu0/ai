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

#include "wareplugin/utils/util.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace wareplugin {

hobot::vision::Points Box2Points(const hobot::vision::BBox &box) {
  hobot::vision::Point top_left = {box.x1, box.y1, box.score};
  hobot::vision::Point bottom_right = {box.x2, box.y2, box.score};
  hobot::vision::Points points;
  points.values = {top_left, bottom_right};
  return points;
}

hobot::vision::BBox Points2Box(const hobot::vision::Points &points) {
  hobot::vision::BBox box;
  box.x1 = points.values[0].x;
  box.y1 = points.values[0].y;
  box.x2 = points.values[1].x;
  box.y2 = points.values[1].y;
  box.score = points.values[0].score;
  return box;
}

}   //  namespace wareplugin
}   //  namespace xproto
}   //  namespace vision
}   //  namespace horizon
