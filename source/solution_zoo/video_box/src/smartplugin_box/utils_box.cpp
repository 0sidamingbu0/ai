/**
 * Copyright (c) 2019 Horizon Robotics. All rights reserved.
 * @file util.h
 * @brief
 * @author ruoting.ding
 * @email ruoting.ding@horizon.ai
 * @date 2019/4/29
 */

#include "smartplugin_box/utils_box.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hobotlog/hobotlog.hpp"
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type.hpp"
namespace horizon {
namespace vision {
namespace utilbox {

void CreateVisionBox(xstream::BaseDataPtr base, HorizonVisionBBox *vision_box) {
  XRocBBox *xroc_box = dynamic_cast<XRocBBox *>(base.get());
  BoxConversion<HorizonVisionBBox, hobot::vision::BBox>(
      vision_box, xroc_box->value);
}

xstream::BaseDataPtr CreateXRocBBox(const HorizonVisionBBox &vision_box) {
  std::shared_ptr<XRocBBox> bbox(new XRocBBox());
  bbox->type_ = "BBox";
  BoxConversion<hobot::vision::BBox, HorizonVisionBBox>(&bbox->value,
                                                                 vision_box);
  return xstream::BaseDataPtr(bbox);
}

xstream::BaseDataPtr CreateXRocFaceBBox(
    const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->face) {
    return CreateXRocBBox(smart_frame->face->face_rect);
  } else {
    return xstream::BaseDataPtr();
  }
}
xstream::BaseDataPtr CreateXRocHeadBBox(
    const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->face) {
    return CreateXRocBBox(smart_frame->face->head_rect);
  } else {
    return xstream::BaseDataPtr();
  }
}

xstream::BaseDataPtr CreateXRocBodyBBox(
    const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->body) {
    return CreateXRocBBox(smart_frame->body->body_rect);
  } else {
    return xstream::BaseDataPtr();
  }
}

xstream::BaseDataPtr CreateXRocPose(const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->face) {
    auto vision_pose3d = smart_frame->face->pose3d;
    std::shared_ptr<XRocPose3D> pose(new XRocPose3D());
    pose->type_ = "Pose3D";
    PoseConversion<hobot::vision::Pose3D, HorizonVisionPose3D>(
        &pose->value, vision_pose3d);
    return xstream::BaseDataPtr(pose);
  } else {
    return xstream::BaseDataPtr();
  }
}

void CreateVisionPose(xstream::BaseDataPtr base,
                      HorizonVisionPose3D *vision_pose) {
  auto xroc_pose = dynamic_cast<XRocPose3D *>(base.get());
  PoseConversion<HorizonVisionPose3D, hobot::vision::Pose3D>(
      vision_pose, xroc_pose->value);
}

xstream::BaseDataPtr CreateXRocLmk(const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->face && smart_frame->face->landmarks) {
    auto vision_lmk = smart_frame->face->landmarks;
    std::shared_ptr<XRocLandmarks> lmk(new XRocLandmarks());
    lmk->type_ = "Landmarks";
    lmk->value.score = vision_lmk->score;
    for (size_t i = 0; i < vision_lmk->num; ++i) {
      hobot::vision::Point p(vision_lmk->points[i].x, vision_lmk->points[i].y,
                             vision_lmk->points[i].score);
      lmk->value.values.emplace_back(p);
    }
    return xstream::BaseDataPtr(lmk);
  } else {
    return xstream::BaseDataPtr();
  }
}

void ConvertLmks(const hobot::vision::Landmarks &xroc_lmks,
                 HorizonVisionLandmarks *vision_lmk, float w_ratio,
                 float h_ratio) {
  HOBOT_CHECK(vision_lmk);
  vision_lmk->score = xroc_lmks.score;
  vision_lmk->num = xroc_lmks.values.size();
  if (vision_lmk->num > 0) {
    vision_lmk->points = static_cast<HorizonVisionPoint *>(
        std::calloc(vision_lmk->num, sizeof(HorizonVisionPoint)));
    for (size_t i = 0; i < vision_lmk->num; ++i) {
      vision_lmk->points[i].x = xroc_lmks.values[i].x * w_ratio;
      vision_lmk->points[i].y = xroc_lmks.values[i].y * h_ratio;
      vision_lmk->points[i].score = xroc_lmks.values[i].score;
    }
  }
}

void ConvertLmks(HorizonVisionLandmarks *vision_lmk,
                 const HorizonVisionLandmarks &lmk, int32_t w_offset,
                 int32_t h_offset, float w_ratio, float h_ratio) {
  HOBOT_CHECK(vision_lmk);
  vision_lmk->num = lmk.num;
  for (size_t i = 0; i < vision_lmk->num; ++i) {
    vision_lmk->points[i].x = (lmk.points[i].x - w_offset) * w_ratio;
    vision_lmk->points[i].y = (lmk.points[i].y - h_offset) * h_ratio;
  }
}

void CreateVisionLmk(xstream::BaseDataPtr base,
                     HorizonVisionLandmarks *vision_lmk) {
  auto xroc_lmk = dynamic_cast<XRocLandmarks *>(base.get());
  HOBOT_CHECK(xroc_lmk);
  ConvertLmks(xroc_lmk->value, vision_lmk);
}

xstream::BaseDataPtr CreateXRocAge(const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->face) {
    auto vision_age = smart_frame->face->age;
    std::shared_ptr<XRocAge> age(new XRocAge());
    age->type_ = "Age";
    AgeConversion<hobot::vision::Age, HorizonVisionAge>(&age->value,
                                                              vision_age);
    return xstream::BaseDataPtr(age);
  } else {
    return xstream::BaseDataPtr();
  }
}

void CreateVisionAge(xstream::BaseDataPtr base, HorizonVisionAge *vision_age) {
  auto xroc_age = dynamic_cast<XRocAge *>(base.get());
  AgeConversion<HorizonVisionAge, hobot::vision::Age>(vision_age,
                                                            xroc_age->value);
}

xstream::BaseDataPtr CreateXRocAttribute(
    const HorizonVisionAttribute &vision_attr, const std::string &type_name) {
  std::shared_ptr<XRocAttribute> attr(new XRocAttribute());
  attr->type_ = type_name;
  attr->value.value = vision_attr.value;
  attr->value.score = vision_attr.score;
  return xstream::BaseDataPtr(attr);
}

xstream::BaseDataPtr CreateXRocGender(
    const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->face) {
    return CreateXRocAttribute(smart_frame->face->gender, "Gender");
  } else {
    return xstream::BaseDataPtr();
  }
}

void CreateVisionAttribute(xstream::BaseDataPtr base,
                           HorizonVisionAttribute *vision_attr) {
  auto xroc_attr = dynamic_cast<XRocAttribute *>(base.get());
  HOBOT_CHECK(xroc_attr) << "input type is " << base->type_;
  vision_attr->score = xroc_attr->value.score;
  vision_attr->value = xroc_attr->value.value;
}

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

xstream::BaseDataPtr CreateXRocKPS(const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->body) {
    /// \todo
    return xstream::BaseDataPtr();
  } else {
    return xstream::BaseDataPtr();
  }
}

xstream::BaseDataPtr CreateXRocReid(const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->body) {
    ///\ todo
    return xstream::BaseDataPtr();
  } else {
    return xstream::BaseDataPtr();
  }
}

xstream::BaseDataPtr CreateXRocMask(const HorizonVisionSmartData *smart_frame) {
  if (smart_frame && smart_frame->body) {
    /// \todo
    return xstream::BaseDataPtr();
  } else {
    return xstream::BaseDataPtr();
  }
}

void CreateVisionSegmentation(xstream::BaseDataPtr base,
                              HorizonVisionSegmentation *vision_mask) {
  auto xroc_segmentation = dynamic_cast<XRocSegmentation *>(base.get());
  HOBOT_CHECK(xroc_segmentation);
  vision_mask->width = xroc_segmentation->value.width;
  vision_mask->height = xroc_segmentation->value.height;
  vision_mask->num = xroc_segmentation->value.values.size();
  vision_mask->values =
      static_cast<float *>(std::calloc(vision_mask->num, sizeof(float)));
  memcpy(vision_mask->values, xroc_segmentation->value.values.data(),
         vision_mask->num * sizeof(float));
}

}  // namespace utilbox
}  // namespace vision
}  //  end of namespace horizon
