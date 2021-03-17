/**
 * Copyright (c) 2020 Horizon Robotics. All rights reserved.
 * @File: DetectPostProcessor.cc
 * @Brief: definition of the DetectPostProcessor
 * @Author: zhe.sun
 * @Email: zhe.sun@horizon.ai
 * @Date: 2020-08-28 14:28:17
 * @Last Modified by: zhe.sun
 * @Last Modified time: 2020-08-28 16:19:08
 */

#include "PostProcessMethod/PostProcessor/DetectPostProcessor.h"
#include <vector>
#include <memory>
#include <set>
#include "common/util.h"
#include "common/apa_data.h"
#include "PostProcessMethod/PostProcessor/DetectConst.h"
#include "bpu_predict/bpu_predict_extension.h"
#include "bpu_predict/bpu_predict.h"
#include "bpu_predict/bpu_parse_utils.h"
#include "bpu_predict/bpu_parse_utils_extension.h"
#include "hobotxstream/profiler.h"
#include "hobotxsdk/xstream_data.h"
#include "hobotlog/hobotlog.hpp"
#include "horizon/vision_type/vision_type.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace xstream {

int DetectPostProcessor::Init(const std::string &cfg) {
  PostProcessor::Init(cfg);
  auto net_info = config_->GetSubConfig("net_info");
  std::vector<std::shared_ptr<postprocessmethod::Config>> model_out_sequence =
      net_info->GetSubConfigArray("model_out_sequence");

  for (size_t i = 0; i < model_out_sequence.size(); ++i) {
    std::string type_str = model_out_sequence[i]->GetSTDStringValue("type");
    HOBOT_CHECK(!type_str.empty());
    if (str2detect_out_type.find(type_str) ==
        str2detect_out_type.end()) {
      out_level2branch_info_[i].type = DetectBranchOutType::INVALID;
    } else {
      out_level2branch_info_[i].type = str2detect_out_type[type_str];
      out_level2branch_info_[i].name =
          model_out_sequence[i]->GetSTDStringValue("name");
      out_level2branch_info_[i].box_name =
          model_out_sequence[i]->GetSTDStringValue("box_name");
      out_level2branch_info_[i].labels =
          model_out_sequence[i]->GetLabelsMap("labels");
    }
  }

  method_outs_ = config_->GetSTDStringArray("method_outs");
  iou_threshold_ = config_->GetFloatValue("iou_threshold", iou_threshold_);
  pre_nms_top_n_ = config_->GetIntValue("pre_nms_top_n");
  post_nms_top_n_ = config_->GetIntValue("post_nms_top_n");
  box_score_thresh_ = config_->GetFloatValue("box_score_thresh",
                                             box_score_thresh_);
  box_rect_thresh_ = config_->GetFloatValue("box_rect_thresh",
                                            box_rect_thresh_);
  has_rect_ = config_->GetBoolValue("has_rect",
                                    has_rect_);

  corner_score_threshold_ = config_->GetFloatValue("corner_score_threshold",
                                                   corner_score_threshold_);
  mask_confidence_ = config_->GetBoolValue("mask_confidence",
                                           mask_confidence_);
  is_crop_ = config_->GetBoolValue("is_crop", is_crop_);

  feat_stride_ = config_->GetSTDIntArray("feat_stride", feat_stride_);
  return 0;
}

void DetectPostProcessor::GetModelInfo() {
  HOBOT_CHECK(bpu_model_ != nullptr);

  uint32_t output_layer_num = bpu_model_->output_num;
  for (size_t i = 0; i < output_layer_num; ++i) {
    auto &branch_info = out_level2branch_info_[i];
    // get shifts
    // dim of per shape = 4
    int shape_dim = bpu_model_->outputs[i].aligned_shape.ndim;
    HOBOT_CHECK(shape_dim == 4)
        << "shape_dim = " << shape_dim;
    std::vector<int> aligned_dim(shape_dim);
    std::vector<int> real_dim(shape_dim);
    for (int dim = 0; dim < shape_dim; dim++) {
      aligned_dim[dim] = bpu_model_->outputs[i].aligned_shape.d[dim];
      real_dim[dim] = bpu_model_->outputs[i].shape.d[dim];
    }
    branch_info.shifts = bpu_model_->outputs[i].shifts;
    // TODO(zhe.sun) BPU_LAYOUT_NCHW待处理
    HOBOT_CHECK(bpu_model_->outputs[i].aligned_shape.layout == BPU_LAYOUT_NHWC);
    branch_info.aligned_nhwc = aligned_dim;
    branch_info.real_nhwc = real_dim;

    // element_type_bytes
    branch_info.element_type_bytes = 1;
    switch (bpu_model_->outputs[i].data_type) {
      case BPU_TYPE_TENSOR_F32:
      case BPU_TYPE_TENSOR_S32:
      case BPU_TYPE_TENSOR_U32:
        branch_info.element_type_bytes = 4;
      default:
        break;
    }
    // valid_output_size
    branch_info.valid_output_size = 1;
    for (int dim = 0; dim < bpu_model_->outputs[i].shape.ndim; dim++) {
      branch_info.valid_output_size *= bpu_model_->outputs[i].shape.d[dim];
    }
    branch_info.valid_output_size *= branch_info.element_type_bytes;
    // aligned_output_size
    branch_info.aligned_output_size = 1;
    for (int dim = 0; dim < bpu_model_->outputs[i].aligned_shape.ndim; dim++) {
      branch_info.aligned_output_size *=
          bpu_model_->outputs[i].aligned_shape.d[dim];
    }
    branch_info.aligned_output_size *= branch_info.element_type_bytes;
    // output_off
    if (i == 0) {
      branch_info.output_off = 0;
    } else {
      branch_info.output_off = out_level2branch_info_[i-1].output_off +
        out_level2branch_info_[i-1].aligned_output_size;
    }
  }
}

std::vector<BaseDataPtr> DetectPostProcessor::Do(
    const std::vector<BaseDataPtr> &input,
    const xstream::InputParamPtr &param) {
  std::vector<BaseDataPtr> output;
  RunSingleFrame(input, output);
  return output;
}

void DetectPostProcessor::GlobalIOU(
    const BaseDataPtr &method_output,
    std::vector<BBox> &total_boxes_result) {
  auto method_out_slot = std::static_pointer_cast<
      xstream::BaseDataVector>(method_output);
  std::vector<BBox> total_boxes(method_out_slot->datas_.size());
  for (size_t j = 0; j < method_out_slot->datas_.size(); j++) {
    auto box = std::static_pointer_cast<XStreamData<
        hobot::vision::BBox>>(method_out_slot->datas_[j]);
    total_boxes[j] = box->value;
  }
  LocalIOU(total_boxes, total_boxes_result,
           iou_threshold_, post_nms_top_n_, false);
}

void DetectPostProcessor::RunSingleFrame(
    const std::vector<BaseDataPtr> &frame_input,
    std::vector<BaseDataPtr> &frame_output) {
  LOGD << "DetectPostProcessor RunSingleFrame";
  RUN_PROCESS_TIME_PROFILER("DetectPostProcessor RunSingleFrame");
  RUN_FPS_PROFILER("DetectPostProcessor RunSingleFrame");
  //  HOBOT_CHECK(frame_input.size() == 1);
  std::vector<BaseDataPtr> method_output;
  for (size_t out_index = 0; out_index < method_outs_.size(); ++out_index) {
    frame_output.push_back(std::make_shared<xstream::BaseDataVector>());
    method_output.push_back(std::make_shared<xstream::BaseDataVector>());
  }

  auto async_datas = std::make_shared<xstream::BaseDataVector>();
  if (is_crop_) {
    async_datas = std::static_pointer_cast<
        xstream::BaseDataVector>(frame_input[0]);
  } else {
    async_datas->datas_.push_back(frame_input[0]);
  }

  for (const auto & data : async_datas->datas_) {
    auto async_data = std::static_pointer_cast<XStreamData<
        std::shared_ptr<hobot::vision::AsyncData>>>(data);
    RunSingleFrameImpl(data, method_output);
  }
  if (!is_crop_) {
    frame_output = std::move(method_output);
  } else {
    for (size_t i = 0; i < method_output.size(); i++) {
      std::vector<BBox> total_boxes_result;
      GlobalIOU(method_output[i], total_boxes_result);
      auto frame_out_slot = std::static_pointer_cast<
          xstream::BaseDataVector>(frame_output[i]);
      for (const auto box : total_boxes_result) {
        auto xstream_box = std::make_shared<XStreamData<BBox>>();
        xstream_box->value = box;
        xstream_box->value.score = 1;
        frame_out_slot->datas_.push_back(xstream_box);
      }
    }
  }
}

void DetectPostProcessor::RunSingleFrameImpl(const BaseDataPtr frame_input,
                        std::vector<BaseDataPtr> &frame_output) {
  auto async_data = std::static_pointer_cast<XStreamData<
      std::shared_ptr<hobot::vision::AsyncData>>>(frame_input);
  if (async_data->value->bpu_model == nullptr ||
      async_data->value->task_handle == nullptr ||
      async_data->value->output_tensors == nullptr ||
      async_data->value->input_tensors == nullptr) {
    LOGE << "Invalid AsyncData";
    return;
  }
  bpu_boxes_ = std::static_pointer_cast<std::vector<BPU_BBOX>>(
      async_data->value->bpu_boxes);
  bpu_model_ = std::static_pointer_cast<BPU_MODEL_S>(
      async_data->value->bpu_model);
  task_handle_ = std::static_pointer_cast<BPU_TASK_HANDLE>(
      async_data->value->task_handle);
  input_tensors_ = std::static_pointer_cast<std::vector<BPU_TENSOR_S>>(
      async_data->value->input_tensors);
  output_tensors_ = std::static_pointer_cast<std::vector<BPU_TENSOR_S>>(
      async_data->value->output_tensors);
  src_image_width_ = async_data->value->src_image_width;
  src_image_height_ = async_data->value->src_image_height;
  model_input_height_ = async_data->value->model_input_height;
  model_input_width_ = async_data->value->model_input_width;
  // get model info
  if (!is_init_) {
    GetModelInfo();
    is_init_ = true;
  }

  {
    RUN_PROCESS_TIME_PROFILER("DetectPostRunModel");
    RUN_FPS_PROFILER("DetectPostRunModel");

    HB_BPU_waitModelDone(task_handle_.get());
    // release input
    ReleaseTensor(input_tensors_);
    // release BPU_TASK_HANDLE
    HB_BPU_releaseTask(task_handle_.get());
  }

  {
    RUN_PROCESS_TIME_PROFILER("DetectPostProcess");
    RUN_FPS_PROFILER("DetectPostProcess");

    DetectOutMsg det_result;
    PostProcess(det_result);

    // release output_tensors
    ReleaseTensor(output_tensors_);

    CoordinateTransOutMsg(det_result, src_image_width_, src_image_height_,
                          model_input_width_, model_input_height_,
                          bpu_boxes_);

    for (auto &boxes : det_result.boxes) {
      LOGD << boxes.first << ", num: " << boxes.second.size();
      for (auto &box : boxes.second) {
        LOGD << box;
      }
    }
    for (auto &orient_boxes : det_result.orient_boxes) {
      LOGD << orient_boxes.first << ", num: " << orient_boxes.second.size();
      for (auto &orient_box : orient_boxes.second) {
        LOGD << orient_box;
      }
    }

    // convert DetectOutMsg to xstream data structure
    std::map<std::string, std::shared_ptr<BaseDataVector>> xstream_det_result;
    GetResultMsg(det_result, xstream_det_result);

    for (size_t out_index = 0; out_index < method_outs_.size(); ++out_index) {
      if (xstream_det_result[method_outs_[out_index]]) {
        auto frame_out_slot = std::static_pointer_cast<
            xstream::BaseDataVector>(frame_output[out_index]);
        auto det_result_slot = xstream_det_result[method_outs_[out_index]];
        frame_out_slot->datas_.insert(frame_out_slot->datas_.end(),
                                      det_result_slot->datas_.begin(),
                                      det_result_slot->datas_.end());
      }
    }
  }
}

void DetectPostProcessor::PostProcess(DetectOutMsg &det_result) {
  for (size_t out_level = 0; out_level < output_tensors_->size(); ++out_level) {
    const auto &branch_info = out_level2branch_info_[out_level];
    auto out_type = branch_info.type;
    HB_SYS_flushMemCache(&(output_tensors_->at(out_level).data),
                         HB_SYS_MEM_CACHE_INVALIDATE);
    switch (out_type) {
      case DetectBranchOutType::INVALID:
      {
        auto dump_bpu_data = getenv("dump_bpu_outdata");
        if (dump_bpu_data && !strcmp(dump_bpu_data, "ON")) {
          // test dump raw output
          if (branch_info.element_type_bytes == 1) {
            std::fstream outfile;
            int8_t* result = reinterpret_cast<int8_t*>
                (output_tensors_->at(out_level).data.virAddr);
            std::string file_name = "layer.txt" +
                                    std::to_string(static_cast<int>(out_level));
            outfile.open(file_name, std::ios_base::out|std::ios_base::trunc);
            for (int i = 0;
                 i < branch_info.aligned_nhwc[1] * branch_info.aligned_nhwc[2];
                 i++) {
              int index = i * branch_info.aligned_nhwc[3];
              for (int j = 0; j < branch_info.aligned_nhwc[3]; j++) {
                outfile << +(*(result+index+j)) << " ";
              }
              outfile << std::endl;
            }
            outfile.close();
          } else {
            std::fstream outfile;
            int32_t* result = reinterpret_cast<int32_t*>
                (output_tensors_->at(out_level).data.virAddr);
            std::string file_name = "layer.txt" +
                                    std::to_string(static_cast<int>(out_level));
            outfile.open(file_name, std::ios_base::out|std::ios_base::trunc);
            for (int i = 0;
                 i < branch_info.aligned_nhwc[1] * branch_info.aligned_nhwc[2];
                 i++) {
              int index = i * branch_info.aligned_nhwc[3];
              for (int j = 0; j < branch_info.aligned_nhwc[3]; j++) {
                outfile << +(*(result+index+j)) << " ";
              }
              outfile << std::endl;
            }
            outfile.close();
          }
        }
        break;
      }
      case DetectBranchOutType::APABBOX:
        ParseAPADetectionBox(
            output_tensors_->at(out_level).data.virAddr,
            out_level, det_result.boxes[branch_info.name]);
        break;
      case DetectBranchOutType::BBOX:
        // need next layer data, need flush
        HB_SYS_flushMemCache(&(output_tensors_->at(out_level+1).data),
                             HB_SYS_MEM_CACHE_INVALIDATE);
        ParseDetectionBox(
            output_tensors_->at(out_level).data.virAddr, out_level,
            output_tensors_->at(out_level+1).data.virAddr,
            det_result.boxes[branch_info.name]);
        break;
      case DetectBranchOutType::ORIENTBBOX:
        // need next 4 layer data, need flush
        if (!has_rect_) {
          for (int i = 1; i <= 3; i++) {
            HB_SYS_flushMemCache(&(output_tensors_->at(out_level+i).data),
                                 HB_SYS_MEM_CACHE_INVALIDATE);
          }
          ParseOrientBox(
              output_tensors_->at(out_level).data.virAddr,
              output_tensors_->at(out_level+1).data.virAddr,
              output_tensors_->at(out_level+2).data.virAddr,
              output_tensors_->at(out_level+3).data.virAddr,
              out_level, out_level+1, out_level+2, out_level+3,
              det_result.orient_boxes[branch_info.name]);
        } else {
          for (int i = 1; i <= 4; i++) {
            HB_SYS_flushMemCache(&(output_tensors_->at(out_level+i).data),
                                 HB_SYS_MEM_CACHE_INVALIDATE);
          }
          ParseOrientBox(
              output_tensors_->at(out_level).data.virAddr,
              output_tensors_->at(out_level+1).data.virAddr,
              output_tensors_->at(out_level+2).data.virAddr,
              output_tensors_->at(out_level+3).data.virAddr,
              out_level, out_level+1, out_level+2, out_level+3,
              det_result.orient_boxes[branch_info.name],
              has_rect_,
              output_tensors_->at(out_level+4).data.virAddr,
              out_level+4);
        }
        break;
      case DetectBranchOutType::CORNER:
        ParseCorner(output_tensors_->at(out_level).data.virAddr,
                    out_level,
                    det_result.corners[branch_info.name]);
        break;
      case DetectBranchOutType::MASK:
        ParseDetectionMask(
          output_tensors_->at(out_level).data.virAddr,
          out_level, det_result.segmentations[branch_info.name],
          mask_confidence_);
        break;
      case DetectBranchOutType::FCOSBBOX:
        for (size_t i = 1; i < 3 * feat_stride_.size(); i++) {
          HB_SYS_flushMemCache(&(output_tensors_->at(out_level + i).data),
                               HB_SYS_MEM_CACHE_INVALIDATE);
        }
        ParseFCOSDetectBox(*output_tensors_.get(),
            out_level, det_result.boxes[branch_info.name]);
        break;
      default:
        break;
    }
  }
}

// convert DetectOutMsg to xstream data structure
void DetectPostProcessor::GetResultMsg(
    DetectOutMsg &det_result,
    std::map<std::string, std::shared_ptr<BaseDataVector>>
      &xstream_det_result) {
  // box
  for (const auto &boxes : det_result.boxes) {
    xstream_det_result[boxes.first] =
        std::make_shared<xstream::BaseDataVector>();
    xstream_det_result[boxes.first]->name_ = "rcnn_" + boxes.first;
    for (uint i = 0; i < boxes.second.size(); ++i) {
      const auto &box = boxes.second[i];
      auto xstream_box = std::make_shared<XStreamData<BBox>>();
      xstream_box->value = std::move(box);
      xstream_det_result[boxes.first]->datas_.push_back(xstream_box);
    }
  }

  // orient_box
  for (const auto &orient_boxes : det_result.orient_boxes) {
    xstream_det_result[orient_boxes.first] =
        std::make_shared<xstream::BaseDataVector>();
    xstream_det_result[orient_boxes.first]->name_ =
        "rcnn_" + orient_boxes.first;
    for (uint i = 0; i < orient_boxes.second.size(); ++i) {
      const auto &orient_box = orient_boxes.second[i];
      auto xstream_box = std::make_shared<XStreamData<Oriented_BBox>>();
      xstream_box->value = std::move(orient_box);
      xstream_det_result[orient_boxes.first]->datas_.push_back(xstream_box);
    }
  }

  // corner
  for (const auto &corners : det_result.corners) {
    xstream_det_result[corners.first] =
        std::make_shared<xstream::BaseDataVector>();
    xstream_det_result[corners.first]->name_ = "rcnn_" + corners.first;
    for (uint i = 0; i < corners.second.size(); ++i) {
      const auto &corner = corners.second[i];
      auto xstream_corner = std::make_shared<XStreamData<Point>>();
      xstream_corner->value = std::move(corner);
      xstream_det_result[corners.first]->datas_.push_back(xstream_corner);
    }
  }

  // segmentations
  for (const auto &segmentation_vec : det_result.segmentations) {
    xstream_det_result[segmentation_vec.first] =
        std::make_shared<xstream::BaseDataVector>();
    xstream_det_result[segmentation_vec.first]->name_ =
        "rcnn_" + segmentation_vec.first;
    for (auto &segmentation : segmentation_vec.second) {
      auto xstream_segmentation = std::make_shared<XStreamData<Segmentation>>();
      xstream_segmentation->value = std::move(segmentation);
      xstream_det_result[segmentation_vec.first]->datas_.push_back(
          xstream_segmentation);
    }
  }
}

void DetectPostProcessor::CoordinateTransOutMsg(
    DetectOutMsg &det_result,
    int src_image_width, int src_image_height,
    int model_input_width, int model_input_hight,
    std::shared_ptr<std::vector<BPU_BBOX>> bboxes) {
  int x_drift = 0, y_drift = 0;
  if (bboxes && !bboxes->empty()) {
    x_drift = bboxes->at(0).x1;
    y_drift = bboxes->at(0).y1;
  }

  for (auto &boxes : det_result.boxes) {
    for (auto &box : boxes.second) {
      CoordinateTransform(box.x1, box.y1,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);
      CoordinateTransform(box.x2, box.y2,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);

      CoordinateDrift(box.x1, box.y1, x_drift, y_drift);
      CoordinateDrift(box.x2, box.y2, x_drift, y_drift);
    }
  }

  for (auto &landmarks : det_result.landmarks) {
    for (auto &landmark : landmarks.second) {
      for (auto &point : landmark.values) {
        CoordinateTransform(point.x, point.y,
                            src_image_width, src_image_height,
                            model_input_width, model_input_hight);
      }
    }
  }

  for (auto &orient_boxes : det_result.orient_boxes) {
    for (auto &orient_box : orient_boxes.second) {
      CoordinateTransform(orient_box.x1, orient_box.y1,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);
      CoordinateTransform(orient_box.x2, orient_box.y2,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);
      CoordinateTransform(orient_box.x3, orient_box.y3,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);
      CoordinateTransform(orient_box.x4, orient_box.y4,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);
    }
  }

  for (auto &corners : det_result.corners) {
    for (auto &corner : corners.second) {
      CoordinateTransform(corner.x, corner.y,
                          src_image_width, src_image_height,
                          model_input_width, model_input_hight);
    }
  }
}

// branch_num: layer
void DetectPostProcessor::ParseDetectionBox(
    void* result, int branch_num, void* anchor,
    std::vector<BBox> &boxes) {
  LOGD << "Into ParseDetectionBox";
  RUN_PROCESS_TIME_PROFILER("Parse_DetectionBox");
  RUN_FPS_PROFILER("Parse_DetectionBox");
  uint8_t *det_result = reinterpret_cast<uint8_t *>(result);
  uint8_t *det_anchor = reinterpret_cast<uint8_t *>(anchor);

  int32_t each_result_bytes = 16;
  int32_t layer_base = 0;
  const int32_t max_dpp_branch = 5;   // dpp constraint

  struct BPUDetNMSResultRaw {
    int16_t xmin;
    int16_t ymin;
    int16_t xmax;
    int16_t ymax;
    int8_t score;
    uint8_t class_id;
    int16_t padding1;
    int16_t padding2;
    int16_t padding3;
  };
  struct SortStruct {
    BPUDetNMSResultRaw *pos;
    int16_t h, w;          // anchor中心点
    uint8_t detout_layer;  // 输出层索引
  };

  std::vector<std::vector<SortStruct>> layer_bbox(max_dpp_branch);

  // 偏移量是该层对齐大小
  int8_t *raw_result = reinterpret_cast<int8_t *>(det_result);
  int8_t *raw_anchor = reinterpret_cast<int8_t *>(det_anchor);

  uint16_t valid_len = *(reinterpret_cast<uint16_t *>(raw_result));
  int32_t valid_result_cnt = valid_len / each_result_bytes;
  LOGD << "valid_len: " << valid_len;
  LOGD << "valid_result_cnt: " << valid_result_cnt;
  BPUDetNMSResultRaw *curr_layer_result =
    reinterpret_cast<BPUDetNMSResultRaw *>(raw_result + 16);
  BPUDetNMSResultRaw *curr_layer_anchor =
    reinterpret_cast<BPUDetNMSResultRaw *>(raw_anchor + 16);

  // clear cached data
  for (int32_t k = layer_base; k < layer_base + max_dpp_branch; k++) {
    layer_bbox[k].clear();
  }

  for (int32_t k = 0; k < valid_result_cnt; k++) {
    BPUDetNMSResultRaw *curr_result = &curr_layer_result[k];
    BPUDetNMSResultRaw *curr_anchor = &curr_layer_anchor[k];

    SortStruct data;
    data.pos = curr_result;
    data.detout_layer = static_cast<uint8_t>(branch_num);
    data.h = curr_anchor->ymin +
      (curr_anchor->ymax - curr_anchor->ymin) / 2;
    data.w = curr_anchor->xmin +
      (curr_anchor->xmax - curr_anchor->xmin) / 2;
    int8_t branch_id =
      static_cast<int8_t>(curr_anchor->xmax - curr_anchor->xmin) / 2;

    if (branch_id < 0 || branch_id >= static_cast<int>(layer_bbox.size())) {
      LOGD << static_cast<int>(branch_id)
        << " " << static_cast<int>(curr_anchor->xmax)
        << " " << static_cast<int>(curr_anchor->xmin);
      continue;
    }
    layer_bbox[branch_id].push_back(data);
  }

  std::vector<BBox> vect_bbox;
  vect_bbox.clear();

  LOGD << "========== orig dump begin ============";
  for (size_t i = 0; i < layer_bbox.size(); i++) {
    if (layer_bbox[i].empty()) {
      continue;
    }
    std::stable_sort(layer_bbox[i].begin(), layer_bbox[i].end(),
      [](const SortStruct &s1, const SortStruct & s2) {
      if (s1.h == s2.h) {
        return (s1.w < s2.w);
      } else {
        return (s1.h < s2.h);
      }
    });

    for (size_t j = 0; j < layer_bbox[i].size(); j++) {
      const BPUDetNMSResultRaw &curr_result = *layer_bbox[i][j].pos;

      int16_t xmin_i = curr_result.xmin >> 2;
      float xmin_f = static_cast<float>(curr_result.xmin & 0x3) / 4.0f;
      float xmin = static_cast<float>(xmin_i) + xmin_f;
      int16_t ymin_i = curr_result.ymin >> 2;
      float ymin_f = static_cast<float>(curr_result.ymin & 0x3) / 4.0f;
      float ymin = static_cast<float>(ymin_i) + ymin_f;
      int16_t xmax_i = curr_result.xmax >> 2;
      float xmax_f = static_cast<float>(curr_result.xmax & 0x3) / 4.0f;
      float xmax = static_cast<float>(xmax_i) + xmax_f;
      int16_t ymax_i = curr_result.ymax >> 2;
      float ymax_f = static_cast<float>(curr_result.ymax & 0x3) / 4.0f;
      float ymax = static_cast<float>(ymax_i) + ymax_f;

      BBox bbox;
      bbox.x1 = xmin;
      bbox.y1 = ymin;
      bbox.x2 = xmax;
      bbox.y2 = ymax;
      bbox.id = curr_result.class_id;
      bbox.score = curr_result.score;  // conf_scale
      vect_bbox.push_back(bbox);
    }
  }

  LOGD << "========== nms dump begin ============";
  std::vector<BBox> merged_local;
  LocalIOU(vect_bbox, merged_local,
           iou_threshold_, vect_bbox.size(), false);

  for (size_t m = 0; m < merged_local.size(); m++) {
    LOGD << merged_local[m].id << merged_local[m];
    // 存入boxes
    boxes.push_back(merged_local[m]);
  }
}

void DetectPostProcessor::ParseDetectionMask(
    void* result, int branch_num,
    std::vector<Segmentation> &masks,
    bool confidence) {
  RUN_PROCESS_TIME_PROFILER("Parse_Mask");
  RUN_FPS_PROFILER("Parse_Mask");
  masks.resize(1);

  uint8_t* mask_feature = reinterpret_cast<uint8_t *>(result);
  Segmentation &mask = masks[0];
  mask.height = out_level2branch_info_[branch_num].aligned_nhwc[1];
  mask.width = out_level2branch_info_[branch_num].aligned_nhwc[2];
  int feature_size = mask.height * mask.width;
  {
    std::unique_lock<std::mutex> lck(mem_mtx_);
    size_t size = feature_size*sizeof(float);
    if (confidence) {
      size *= 2;
    }
    if (mem_seg_.first == nullptr) {
      mem_seg_.first = new char[size];
      mem_seg_.second = size;
    } else if (mem_seg_.second < size) {
      delete[] mem_seg_.first;
      mem_seg_.first = new char[size];
      mem_seg_.second = size;
    }
  }
  float *mask_values = reinterpret_cast<float *>(mem_seg_.first);
  mask.values = std::vector<float>(mask_values, mask_values+feature_size);

  float score = 0;
  for (int i = 0; i < feature_size; ++i) {
    uint8_t fp_mask = *(mask_feature + 2*i);
    mask.values[i] = fp_mask;
  }
  if (confidence) {
    mask.pixel_score = std::vector<float>(
        mask_values+feature_size, mask_values+feature_size*2);
    for (int i = 0; i < feature_size; ++i) {
      uint8_t fp_score = *(mask_feature + 2*i + 1);
      mask.pixel_score[i] = fp_score;
      score += fp_score;
    }
    score = score / feature_size;
    mask.score = score;
  }
  #if 0
  // test mask
  cv::Mat mask_cv(mask.height, mask.width, CV_8UC1);
  for (int h = 0; h < mask.height; ++h) {
    uchar *p_gray = mask_cv.ptr<uchar>(h);
    for (int w = 0; w < mask.width; ++w) {
      p_gray[w] = mask.values[h*mask.width+w];
    }
  }
  cv::imwrite("mask.jpg", mask_cv);
  #endif
}

void DetectPostProcessor::ParseAPADetectionBox(
    void* result, int branch_num,
    std::vector<BBox> &boxes) {
  LOGD << "Into ParseAPADetectionBox";
  RUN_PROCESS_TIME_PROFILER("Parse_APADetectionBox");
  RUN_FPS_PROFILER("Parse_APADetectionBox");
  int8_t *anchor_result = reinterpret_cast<int8_t *>(result);
  std::vector<BPUParkingAnchorResult> cached_result;

  std::vector<float> anchor_scale(
      out_level2branch_info_[branch_num].aligned_nhwc[3],
      32);

  int dim_h = out_level2branch_info_[branch_num].aligned_nhwc[1];
  int dim_w = out_level2branch_info_[branch_num].aligned_nhwc[2];
  int dim_c = out_level2branch_info_[branch_num].aligned_nhwc[3];
  int line_step = dim_w * dim_c;
  // 1. find max conf point and get anchor id and type
  {
    for (int h = 0; h < dim_h; ++h) {
      int8_t * conf_index = anchor_result + h * line_step;
      for (int w = 0; w < dim_w; w++) {
        for (int channel_id = 0; channel_id < 2; channel_id++) {
          int conf = *(conf_index + w * dim_c + channel_id * 6);
          // conf 阈值
          if (conf >= -0.8473 * anchor_scale[channel_id]) {
            BPUParkingAnchorResult result;
            result.h = h;
            result.w = w;
            result.anchor = channel_id;
            result.conf = conf;
            result.c = channel_id;
            cached_result.push_back(result);
          }
        }
      }
    }
  }

  // 2 get anchor
  {
    for (size_t i = 0; i < cached_result.size(); ++i) {
      BPUParkingAnchorResult &result =
          cached_result[i];
      int anchor_id = result.anchor;
      int8_t *base_addr =
        anchor_result + result.h * line_step +
        result.w * dim_c + anchor_id * 6 + 1;    // dx

      result.box.data.x = *base_addr;
      result.box.data.y = *(base_addr + 1);
      result.box.data.z = 0;
      result.box.data.l = *(base_addr + 2);
      result.box.data.w = *(base_addr + 3);
      result.box.data.h = 0;
      result.box.data.theta = *(base_addr + 4);
    }
  }

  // 3 decode 2DBoxTo3D
  std::vector<Parking3DBBox> result_box;
  {
    float da = 0.0f, xa = 0.0f, ya = 0.0f;
    float la = 0.0f, wa = 0.0f, thetaa = 0.0f;
    float ouput_scale = 8;  // TODO(zhe.sun) need get from modelinfo
    for (size_t k = 0; k < cached_result.size(); k++) {
      BPUParkingAnchorResult &unet_obj = cached_result[k];
      int anchor_id = unet_obj.anchor;
      int c = unet_obj.c;

      std::vector<LidarAnchor> lidar_anchor {
          {0, 0, 0, 100, 50, 0, 0, 111.803},
          {0, 0, 0, 100, 50, 0, 1.57, 111.803}};
      xa = lidar_anchor[anchor_id].x + unet_obj.w * ouput_scale;
      ya = lidar_anchor[anchor_id].y + unet_obj.h * ouput_scale;
      la = lidar_anchor[anchor_id].l;
      wa = lidar_anchor[anchor_id].w;
      thetaa = lidar_anchor[anchor_id].t;
      da = lidar_anchor[anchor_id].d;
      float y_min_ = -1000.0f;
      float y_max_ = 5750.0f;
      float x_min_ = -1000.0f;
      float x_max_ = 5750.0f;
      float center_x, center_y, result_l, result_w, theta;
      center_x = (unet_obj.box.data.x / anchor_scale[0]) * da + xa;
      if (center_x > x_max_ || center_x < x_min_) {
        continue;
      }
      center_y = (unet_obj.box.data.y / anchor_scale[1]) * da + ya;
      if (center_y > y_max_ || center_y < y_min_) {
        continue;
      }
      result_l = expf(unet_obj.box.data.l / anchor_scale[3]) * la;
      result_w = expf(unet_obj.box.data.w / anchor_scale[4]) * wa;
      theta = unet_obj.box.data.theta / anchor_scale[6] + thetaa;
      theta = -theta;

      Parking3DBBox box(unet_obj.conf / anchor_scale[c],
          center_x, center_y, 0, result_l, 0,
          result_w, theta);
      result_box.push_back(box);
    }
  }

  // 4. NMS
  {
    std::vector<Parking3DBBox> merge_result;
    float nms_thresh_hold = 0.3;
    Parking_NMS_local_iou(result_box, merge_result, nms_thresh_hold,
                          result_box.size(), false);
    std::vector<cv::Mat> out_2dbox;
    Center_To_Corner_Box2d(merge_result, out_2dbox);
    LOGD << "result size: " << merge_result.size();
    for (size_t j = 0; j < merge_result.size(); ++j) {
      LOGD << "x: " << merge_result[j].x
           << ", y: " << merge_result[j].y
           << ", width: " << merge_result[j].w
           << ", length: " << merge_result[j].l
           << ", theta: " << merge_result[j].theta;
      // BBox
      BBox box;
      {
        box.x1 = merge_result[j].x - merge_result[j].l / 2;
        box.y1 = merge_result[j].y - merge_result[j].w / 2;
        box.x2 = merge_result[j].x + merge_result[j].l / 2;
        box.y2 = merge_result[j].y + merge_result[j].w / 2;
        box.score = merge_result[j].score;
        box.rotation_angle = merge_result[j].theta;
      }
      boxes.push_back(box);
    }
  }
}

void DetectPostProcessor::LocalIOU(
    std::vector<BBox> &candidates,
    std::vector<BBox> &result,
    const float overlap_ratio, const int top_N,
    const bool addScore) {
  if (candidates.size() == 0) {
    return;
  }
  std::vector<bool> skip(candidates.size(), false);

  auto greater = [](const BBox &a, const BBox &b) {
    return a.score > b.score;
  };
  std::stable_sort(candidates.begin(), candidates.end(),
                   greater);

  int count = 0;
  for (size_t i = 0; count < top_N && i < skip.size(); ++i) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    float area_i = candidates[i].Width() * candidates[i].Height();

    // suppress the significantly covered bbox
    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      // get intersections
      float xx1 =
          std::max(candidates[i].x1, candidates[j].x1);
      float yy1 =
          std::max(candidates[i].y1, candidates[j].y1);
      float xx2 =
          std::min(candidates[i].x2, candidates[j].x2);
      float yy2 =
          std::min(candidates[i].y2, candidates[j].y2);
      float area_intersection = (xx2 - xx1) * (yy2 - yy1);
      bool area_intersection_valid = (area_intersection > 0) && (xx2 - xx1 > 0);

      if (area_intersection_valid) {
        // compute overlap
        float area_j = candidates[j].Width() * candidates[j].Height();
        float o = area_intersection / (area_i + area_j - area_intersection);

        if (o > overlap_ratio) {
          skip[j] = true;
          if (addScore) {
            candidates[i].score += candidates[j].score;
          }
        }
      }
    }
    result.push_back(candidates[i]);
  }
  return;
}

void DetectPostProcessor::ComputeLocations(
    const int &height,
    const int &width,
    const int &feature_stride,
    cv::Mat &location) {
  std::vector<int32_t> shift_x = Arange(0, width);
  std::vector<int32_t> shift_y = Arange(0, height);
  cv::Mat shift_x_mat = cv::Mat(shift_x) * feature_stride;
  cv::Mat shift_y_mat = cv::Mat(shift_y) * feature_stride;
  // meshgrid
  cv::Mat shift_x_mesh, shift_y_mesh;
  MeshGrid(shift_x_mat, shift_y_mat, shift_x_mesh, shift_y_mesh);
  cv::Mat concat_xy;
  cv::vconcat(shift_x_mesh.reshape(1, 1),
              shift_y_mesh.reshape(1, 1),
              concat_xy);    // 纵向拼接两个行向量
  cv::transpose(concat_xy, location);
  location = location + feature_stride / 2;  // 向下取整
}

void DetectPostProcessor::ParseFCOSDetectBox(
    const std::vector<BPU_TENSOR_S> &output_tensors,
    int branch_level, std::vector<BBox> &result_boxes) {
  RUN_PROCESS_TIME_PROFILER("Parse_FCOS_Detect_Box")
  RUN_FPS_PROFILER("Parse_FCOS_Detect_Box")

  int height = out_level2branch_info_[branch_level].real_nhwc[1];
  int width = out_level2branch_info_[branch_level].real_nhwc[2];
  const auto stride_size = feat_stride_.size();

  std::vector<void*> bpu_out_scores(stride_size);
  std::vector<void*> bpu_out_regs(stride_size);
  std::vector<void*> bpu_out_ctrs(stride_size);

  for (size_t i = 0; i < stride_size; i++) {
    auto feat_offset = branch_level + i;
    bpu_out_scores[i] = output_tensors[feat_offset].data.virAddr;
    bpu_out_regs[i] = output_tensors[feat_offset + stride_size].data.virAddr;
    bpu_out_ctrs[i] = output_tensors[
        feat_offset + stride_size * 2].data.virAddr;
  }

  // 1. 计算fmap上每个点对应其在原图中的坐标locations : (h*w, 2)
  static std::vector<cv::Mat> locations(stride_size);  // 对应feat_stride
  static std::once_flag flag;
  std::call_once(flag, [&width, &height, this]() {
    for (size_t i = 0; i < locations.size(); i++) {
      int scale = 1 << i;
      int height_scale = height / scale;
      int width_scale = width / scale;
      ComputeLocations(height_scale, width_scale,
                       feat_stride_[i], locations[i]);
    }
  });

  // 2. 根据定点阈值，转换定点数据
  // scores_valid_index[i]: stride i
  std::vector<std::vector<int>> scores_valid_index(stride_size);
  std::vector<std::vector<int32_t>> scores_valid_float(stride_size);
  std::vector<std::vector<float>> reg_valid_float(stride_size);
  std::vector<std::vector<int32_t>> ctr_valid_float(stride_size);


  // 2.1 先解析score, 过滤得到初步的valid_index
  float box_score_tmp_thresh =
      log(box_score_thresh_ / (1.0 - box_score_thresh_));

  std::vector<int> valid_nums(stride_size);

  for (size_t i = 0; i < stride_size; i++) {
    scores_valid_float[i].resize(height * width);
    scores_valid_index[i].resize(height * width);
    ConvertOutputFilter(
        bpu_out_scores[i],
        scores_valid_float[i].data(),
        branch_level + i,
        box_score_tmp_thresh,
        scores_valid_index[i], valid_nums[i]);
    LOGD << "valid_num: " << valid_nums[i];
  }

  for (size_t i = 0; i < stride_size; i++) {
    ctr_valid_float[i].resize(valid_nums[i]);
    ConvertOutputFilter(
        bpu_out_ctrs[i],
        ctr_valid_float[i].data(),
        branch_level + i + stride_size * 2,
        scores_valid_index[i], valid_nums[i]);
  }

  for (size_t i = 0; i < stride_size; i++) {
    reg_valid_float[i].resize(valid_nums[i] * 4);
    ConvertOutputFilter(
        bpu_out_regs[i],
        reg_valid_float[i].data(),
        branch_level + i + stride_size,
        scores_valid_index[i], valid_nums[i]);
  }

    std::vector<BBox> box_data;
    int pre_nms_top_n = pre_nms_top_n_;
    for (size_t i = 0; i < stride_size; i++) {  // stride i
      int valid_num = valid_nums[i];
      if (valid_num == 0) continue;
      // score
      cv::Mat box_score_data_raw(valid_num, 1, CV_32FC1,
                                 scores_valid_float[i].data());
      cv::Mat box_ctr_data_raw(valid_num, 1, CV_32FC1,
                               ctr_valid_float[i].data());
      cv::Mat box_score_data, box_ctr_data;
      cv::exp(box_score_data_raw * (-1), box_score_data);
      box_score_data = 1 / (1 + box_score_data);
      cv::exp(box_ctr_data_raw * (-1), box_ctr_data);
      box_ctr_data = 1 / (1 + box_ctr_data);
      cv::multiply(box_score_data, box_ctr_data, box_score_data);
      // reg
      cv::Mat box_reg_data_raw(valid_num, 4, CV_32FC1,
                           reg_valid_float[i].data());
      cv::Mat box_reg_data;
      cv::exp(box_reg_data_raw, box_reg_data);
      box_reg_data = box_reg_data * feat_stride_[i];

      // box_score_thresh_过滤
      cv::Mat candidate_inds;
      // 元素值大于thresh 得到255，否则0
      cv::compare(
          box_score_data, box_score_thresh_, candidate_inds, cv::CMP_GT);

      candidate_inds = candidate_inds / 255;  // 0 or 1
      pre_nms_top_n = cv::sum(candidate_inds)[0];  // box_num
      LOGD << "pre_nms_top_n: " << pre_nms_top_n
           << " , pre_nms_top_n_: " << pre_nms_top_n_;

      // 根据score阈值过滤
      for (int row = 0; row < box_score_data.rows; row++) {
        float* ptr = box_score_data.ptr<float>(row);
        for (int col = 0; col < box_score_data.cols; col++) {
          if (ptr[col] > box_score_thresh_) {
            BBox one_box_data;
            one_box_data.score = ptr[col];
            int location_0, location_1;
            location_0 = locations[i].ptr<int32_t>(
                scores_valid_index[i][row])[0];
            location_1 = locations[i].ptr<int32_t>(
                scores_valid_index[i][row])[1];

            // 限制框大小在图像内
            one_box_data.x1 = std::max(
                std::min(location_0 - box_reg_data.ptr<float>(row)[0],
                         static_cast<float>(model_input_width_) - 1), 0.0f);
            one_box_data.y1 = std::max(
                std::min(location_1 - box_reg_data.ptr<float>(row)[1],
                         static_cast<float>(model_input_height_) - 1), 0.0f);
            one_box_data.x2 = std::max(
                std::min(location_0 + box_reg_data.ptr<float>(row)[2],
                         static_cast<float>(model_input_width_) - 1), 0.0f);
            one_box_data.y2 = std::max(
                std::min(location_1 + box_reg_data.ptr<float>(row)[3],
                         static_cast<float>(model_input_height_) - 1), 0.0f);
            box_data.push_back(std::move(one_box_data));
          }
        }
      }
    }  // end stride i
    // NMS box
    if (pre_nms_top_n > pre_nms_top_n_) {  // 需要取前top_n_个
      auto greater = [](const BBox &a, const BBox &b) {
        return a.score > b.score;
      };
      std::sort(box_data.begin(), box_data.end(), greater);  // 降序排序
      box_data.resize(pre_nms_top_n_);
    }

    LocalIOU(box_data, result_boxes,
        iou_threshold_, post_nms_top_n_, false);
}

// branch 所在层
void DetectPostProcessor::ParseOrientBox(
    void* box_score, void* box_reg, void* box_ctr, void* box_orient,
    int branch_score, int branch_reg, int branch_ctr, int branch_orient,
    std::vector<Oriented_BBox> &oriented_boxes,
    bool has_rect, void* box_rect, int branch_rect) {
  RUN_PROCESS_TIME_PROFILER("Parse_Orient_Box");
  int height = out_level2branch_info_[branch_score].real_nhwc[1];
  int width = out_level2branch_info_[branch_score].real_nhwc[2];
  int stride = out_level2branch_info_[branch_score].aligned_nhwc[3];

  // 1. 计算fmap上每个点对应其在原图中的坐标locations : (h*w, 2)
  static cv::Mat locations;
  static std::once_flag flag;
  std::call_once(flag, [&width, &height, &stride]() {
    std::vector<int32_t> shift_x = Arange(0, width);
    std::vector<int32_t> shift_y = Arange(0, height);
    cv::Mat shift_x_mat = cv::Mat(shift_x) * stride;
    cv::Mat shift_y_mat = cv::Mat(shift_y) * stride;
    // meshgrid
    cv::Mat shift_x_mesh, shift_y_mesh;
    MeshGrid(shift_x_mat, shift_y_mat, shift_x_mesh, shift_y_mesh);
    cv::Mat concat_xy;
    cv::vconcat(shift_x_mesh.reshape(1, 1),
                shift_y_mesh.reshape(1, 1),
                concat_xy);    // 纵向拼接两个行向量
    cv::transpose(concat_xy, locations);
    locations = locations + stride / 2;  // 向下取整
  });

  // 2. 准备数据 维度[n, h*w, c] n=1
  struct BPU_Data {
    void* data;
    int branch;
  };
  std::vector<BPU_Data> bpu_datas =
    {{box_score, branch_score},
     {box_reg, branch_reg},
     {box_ctr, branch_ctr},
     {box_orient, branch_orient}};

  // 2.1. 先解析score, 过滤得到初步的valid_index
  int model_out_size = height * width;
  std::vector<float> vec_data_score(model_out_size);
  std::vector<int> valid_index(model_out_size);
  int valid_num = 0;
  ConvertOutputFilter(bpu_datas[0].data,
                      vec_data_score.data(),
                      bpu_datas[0].branch,
                      log(box_score_thresh_/(1-box_score_thresh_)),
                      valid_index,
                      valid_num);
  LOGD << "valid_num: " << valid_num;
  if (valid_num <= 0) {
    LOGD << "no box detected";
    return;
  }

  // 2.2. 解析ctr、reg、ori, 只需根据valid_index解析部分即可
  std::vector<float> vec_data_ctr(valid_num);
  ConvertOutputFilter(bpu_datas[2].data,
                      vec_data_ctr.data(),
                      bpu_datas[2].branch,
                      valid_index,
                      valid_num);
  std::vector<float> vec_data_reg(valid_num * 4);
  ConvertOutputFilter(bpu_datas[1].data,
                      vec_data_reg.data(),
                      bpu_datas[1].branch,
                      valid_index,
                      valid_num);
  std::vector<float> vec_data_ori(valid_num * 2);
  ConvertOutputFilter(bpu_datas[3].data,
                      vec_data_ori.data(),
                      bpu_datas[3].branch,
                      valid_index,
                      valid_num);
  cv::Mat box_rect_candidate;
  if (has_rect) {
    std::vector<float> vec_data_rect(valid_num);
    ConvertOutputFilter(box_rect,
                        vec_data_rect.data(),
                        branch_rect,
                        valid_index,
                        valid_num);
    // 计算rect
    cv::Mat box_rect_data;
    cv::Mat box_rect_data_raw(valid_num, 1, CV_32FC1, vec_data_rect.data());
    cv::exp(box_rect_data_raw * (-1), box_rect_data);
    box_rect_data = 1 / (1 + box_rect_data);
    // 元素值大于thresh 得到255，否则0
    cv::compare(box_rect_data, box_rect_thresh_,
                box_rect_candidate, cv::CMP_GT);
    box_rect_candidate /= 255;
  }
  // 2.3. 计算score
  cv::Mat box_score_data_raw(valid_num, 1, CV_32FC1, vec_data_score.data());
  cv::Mat box_ctr_data_raw(valid_num, 1, CV_32FC1, vec_data_ctr.data());
  cv::Mat box_score_data, box_ctr_data;
  cv::exp(box_score_data_raw * (-1), box_score_data);
  box_score_data = 1 / (1 + box_score_data);
  cv::exp(box_ctr_data_raw * (-1), box_ctr_data);
  box_ctr_data = 1 / (1 + box_ctr_data);
  cv::multiply(box_score_data, box_ctr_data, box_score_data);
  // 计算reg
  cv::Mat box_reg_data_raw(valid_num, 4, CV_32FC1, vec_data_reg.data());
  cv::Mat box_reg_data;
  cv::exp(box_reg_data_raw, box_reg_data);
  box_reg_data = box_reg_data * 8;
  // 计算ori
  cv::Mat box_orient_data(valid_num, 2, CV_32FC1, vec_data_ori.data());

  // box_score_thresh_过滤
  cv::Mat candidate_inds;
  // 元素值大于thresh 得到255，否则0
  cv::compare(box_score_data, box_score_thresh_, candidate_inds, cv::CMP_GT);

  candidate_inds = candidate_inds / 255;  // 0 or 1
  int pre_nms_top_n = cv::sum(candidate_inds)[0];  // box_num
  LOGD << "pre_nms_top_n: " << pre_nms_top_n
       << " , pre_nms_top_n_: " << pre_nms_top_n_;
  struct BoxData {
    float score;
    int class_id;
    std::vector<int> location;  // size: 2
    std::vector<float> reg;       // size: 4
    std::vector<float> orient;    // size: 2
  };
  std::vector<BoxData> box_data(pre_nms_top_n);
  int box_data_index = 0;
  // [valid_num, c]
  for (int row = 0; row < box_score_data.rows; row++) {
    float* ptr = box_score_data.ptr<float>(row);
    for (int col = 0; col < box_score_data.cols; col++) {
      if (ptr[col] > box_score_thresh_) {
        BoxData one_box_data;
        one_box_data.score = ptr[col];
        std::vector<int> location(2);
        std::vector<float> reg(4), orient(2);
        location[0] = locations.ptr<int32_t>(valid_index[row])[0];
        location[1] = locations.ptr<int32_t>(valid_index[row])[1];
        reg[0] = box_reg_data.ptr<float>(row)[0];
        reg[1] = box_reg_data.ptr<float>(row)[1];
        reg[2] = box_reg_data.ptr<float>(row)[2];
        reg[3] = box_reg_data.ptr<float>(row)[3];

        orient[0] =
          std::max(std::min(box_orient_data.ptr<float>(row)[0], 2.0f), 0.0f);
        orient[1] =
          std::max(std::min(box_orient_data.ptr<float>(row)[1], 2.0f), 0.0f);
        one_box_data.location = location;
        one_box_data.reg = reg;
        one_box_data.orient = orient;

        if (has_rect) {
          // class_id表示矩形框与否
          one_box_data.class_id = box_rect_candidate.ptr<uchar>(row)[0];
        } else {
          one_box_data.class_id = col + 1;
        }

        box_data[box_data_index++] = one_box_data;
      }
    }
  }
  if (pre_nms_top_n > pre_nms_top_n_) {  // 需要取前top_n_个
    pre_nms_top_n = pre_nms_top_n_;
    auto greater = [](const BoxData &a, const BoxData &b) {
      return a.score > b.score;
    };
    std::sort(box_data.begin(), box_data.end(), greater);  // 降序排序
  }

  // 3. 根据HBB获取obb
  std::vector<std::vector<float>> horizontal_boxes(pre_nms_top_n);
  std::vector<float> per_box_w_rot(pre_nms_top_n), per_box_h_rot(pre_nms_top_n);
  // obb: non-nms
  std::vector<Oriented_BBox> obb(pre_nms_top_n);
  for (int i = 0; i < pre_nms_top_n; i++) {
    horizontal_boxes[i].push_back(box_data[i].location[0] - box_data[i].reg[0]);
    horizontal_boxes[i].push_back(box_data[i].location[1] - box_data[i].reg[1]);
    horizontal_boxes[i].push_back(box_data[i].location[0] + box_data[i].reg[2]);
    horizontal_boxes[i].push_back(box_data[i].location[1] + box_data[i].reg[3]);
    per_box_w_rot[i] =
      (box_data[i].reg[2] + box_data[i].reg[0]) / 2 * box_data[i].orient[0];
    per_box_h_rot[i] =
      (box_data[i].reg[3] + box_data[i].reg[1]) / 2 * box_data[i].orient[1];

    obb[i].x1 = horizontal_boxes[i][0];
    obb[i].y1 = horizontal_boxes[i][3] - per_box_h_rot[i];
    obb[i].x2 = horizontal_boxes[i][2] - per_box_w_rot[i];
    obb[i].y2 = horizontal_boxes[i][1];
    obb[i].x3 = horizontal_boxes[i][2];
    obb[i].y3 = horizontal_boxes[i][1] + per_box_h_rot[i];
    obb[i].x4 = horizontal_boxes[i][0] + per_box_w_rot[i];
    obb[i].y4 = horizontal_boxes[i][3];

    obb[i].score = box_data[i].score;
    obb[i].id = box_data[i].class_id;
  }

  // 4. 对obb进行NMS
  LocalIOU(obb, oriented_boxes, iou_threshold_, post_nms_top_n_, false);
  return;
}

void DetectPostProcessor::LocalIOU(
    std::vector<Oriented_BBox> &candidates,
    std::vector<Oriented_BBox> &result,
    const float overlap_ratio, const int top_N,
    const bool addScore) {
  if (candidates.size() == 0) {
    return;
  }
  std::vector<bool> skip(candidates.size(), false);

  auto greater = [](const Oriented_BBox &a, const Oriented_BBox &b) {
    return a.score > b.score;
  };
  std::stable_sort(candidates.begin(), candidates.end(),
                   greater);

  int count = 0;
  for (size_t i = 0; count < top_N && i < skip.size(); ++i) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    float length_i_1 = candidates[i].Length1();
    float length_i_2 = candidates[i].Length2();
    int width_i = std::min(length_i_1, length_i_2);
    int height_i = std::max(length_i_1, length_i_2);
    float area_i = width_i * height_i;
    cv::RotatedRect rect_i = cv::RotatedRect(
        cv::Point2f(candidates[i].CenterX(), candidates[i].CenterY()),
        cv::Size2f(height_i, width_i),
        candidates[i].Theta());

    // suppress the significantly covered bbox
    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      // get intersections
      float length_j_1 = candidates[j].Length1();
      float length_j_2 = candidates[j].Length2();
      float width_j = std::min(length_j_1, length_j_2);
      float height_j = std::max(length_j_1, length_j_2);
      cv::RotatedRect rect_j = cv::RotatedRect(
          cv::Point2f(candidates[j].CenterX(), candidates[j].CenterY()),
          cv::Size2f(height_j, width_j),
          candidates[j].Theta());
      std::vector<cv::Point2f> vertices;
      cv::rotatedRectangleIntersection(rect_i, rect_j, vertices);
      if (vertices.size() > 0) {
        // compute overlap
        float area_intersection = cv::contourArea(vertices);
        float area_j = width_j * height_j;
        float o = area_intersection / (area_i + area_j - area_intersection);

        if (o > overlap_ratio) {
          skip[j] = true;
          if (addScore) {
            candidates[i].score += candidates[j].score;
          }
        }
      }
    }
    result.push_back(candidates[i]);
  }
  return;
}

void DetectPostProcessor::ParseCorner(
    void* result, int branch,
    std::vector<Point> &corners) {
  RUN_PROCESS_TIME_PROFILER("Parse_Corner");
  RUN_FPS_PROFILER("Parse_Corner");
  int height = out_level2branch_info_[branch].real_nhwc[1];
  int width = out_level2branch_info_[branch].real_nhwc[2];
  // int channel = out_level2branch_info_[branch].real_nhwc[3];
  int model_out_size = height * width * 1;

  std::vector<float> vec_data(model_out_size);
  ConvertOutput(result, vec_data.data(), branch, 1);  // 暂时只用到了第一个通道

  // get first channel: heatmap
  cv::Mat heatmap(height, width, CV_32FC(1), vec_data.data());

  // 1. find local_max
  std::vector<float> scores;
  std::vector<std::pair<int, int>> points;
  for (int row = 0; row < heatmap.rows; row++) {
    for (int col = 0; col < heatmap.cols; col++) {
      float* ptr = heatmap.ptr<float>(row);
      if (ptr[col] < corner_score_threshold_) {
        continue;
      }
      if (row > 0 && ptr[col] < heatmap.ptr<float>(row-1)[col]) {
        continue;
      }
      if (row < heatmap.rows-1 && ptr[col] <= heatmap.ptr<float>(row+1)[col]) {
        continue;
      }
      if (col > 0 && ptr[col] < ptr[col-1]) {
        continue;
      }
      if (col < heatmap.cols-1 && ptr[col] <= ptr[col+1]) {
        continue;
      }
      scores.push_back(ptr[col]);
      points.push_back(std::make_pair(col, row));
    }
  }

  // 2. find bias
  int nums = points.size();
  std::vector<std::pair<float, float>> bias(nums);
  for (int i = 0; i < nums; i++) {
    int x = points[i].first;
    int y = points[i].second;
    int diff_x = 0, diff_y = 0;
    if (x > 0 && x < heatmap.cols-1) {
      diff_x = heatmap.ptr<float>(y)[x+1] >
          heatmap.ptr<float>(y)[x-1] ? 1 : -1;
    }
    if (y > 0 && y < heatmap.rows-1) {
      diff_y = heatmap.ptr<float>(y+1)[x] >
          heatmap.ptr<float>(y-1)[x] ? 1 : -1;
    }
    float bias_x = diff_x * 0.25;
    float bias_y = diff_y * 0.25;
    bias[i] = std::make_pair(bias_x, bias_y);
  }

  // 3. (max + bias) * stride = location
  corners.resize(nums);
  for (int i = 0; i < nums; i++) {
    corners[i].x = (points[i].first + bias[i].first) * 4;
    corners[i].y = (points[i].second + bias[i].second) * 4;
    corners[i].score = scores[i];
    LOGD << "x: " << corners[i].x
         << ", y: " << corners[i].y
         << ", score: " << corners[i].score;
  }
}

void DetectPostProcessor::ConvertOutput(
    void *src_ptr,
    void *dest_ptr,
    int out_index,
    int channel) {
  auto &aligned_shape = bpu_model_->outputs[out_index].aligned_shape;
  auto &real_shape = bpu_model_->outputs[out_index].shape;
  auto elem_size = 4;
  HOBOT_CHECK(bpu_model_->outputs[out_index].data_type == BPU_TYPE_TENSOR_S32);
  auto shift = bpu_model_->outputs[out_index].shifts;

  int convert_channel = 0;
  if (channel == 0) {
    convert_channel = real_shape.d[3];
  } else {
    HOBOT_CHECK(channel <= real_shape.d[3]) << "channel: " << channel
        << ", real_channel: " << real_shape.d[3];
    convert_channel = channel;
  }

  uint32_t dst_n_stride =
      real_shape.d[1] * real_shape.d[2] * convert_channel * elem_size;
  uint32_t dst_h_stride = real_shape.d[2] * convert_channel * elem_size;
  uint32_t dst_w_stride = convert_channel * elem_size;
  uint32_t src_n_stride =
      aligned_shape.d[1] * aligned_shape.d[2] * aligned_shape.d[3] * elem_size;
  uint32_t src_h_stride = aligned_shape.d[2] * aligned_shape.d[3] * elem_size;
  uint32_t src_w_stride = aligned_shape.d[3] * elem_size;

  float tmp_float_value;
  int32_t tmp_int32_value;

  for (int nn = 0; nn < real_shape.d[0]; nn++) {
    void *cur_n_dst = reinterpret_cast<int8_t *>(dest_ptr) + nn * dst_n_stride;
    void *cur_n_src = reinterpret_cast<int8_t *>(src_ptr) + nn * src_n_stride;
    for (int hh = 0; hh < real_shape.d[1]; hh++) {
      void *cur_h_dst =
          reinterpret_cast<int8_t *>(cur_n_dst) + hh * dst_h_stride;
      void *cur_h_src =
          reinterpret_cast<int8_t *>(cur_n_src) + hh * src_h_stride;
      for (int ww = 0; ww < real_shape.d[2]; ww++) {
        void *cur_w_dst =
            reinterpret_cast<int8_t *>(cur_h_dst) + ww * dst_w_stride;
        void *cur_w_src =
            reinterpret_cast<int8_t *>(cur_h_src) + ww * src_w_stride;
        for (int cc = 0; cc < convert_channel; cc++) {
          void *cur_c_dst =
              reinterpret_cast<int8_t *>(cur_w_dst) + cc * elem_size;
          void *cur_c_src =
              reinterpret_cast<int8_t *>(cur_w_src) + cc * elem_size;
          tmp_int32_value = *(reinterpret_cast<int32_t *>(cur_c_src));
          tmp_float_value = GetFloatByInt(tmp_int32_value, shift[cc]);
          *(reinterpret_cast<float *>(cur_c_dst)) = tmp_float_value;
        }
      }
    }
  }
}

void DetectPostProcessor::ConvertOutputFilter(
    void *src_ptr,
    void *dest_ptr,
    int out_index,
    float threshold,
    std::vector<int> &valid_index,
    int &valid_num) {
  auto &aligned_shape = bpu_model_->outputs[out_index].aligned_shape;
  auto &real_shape = bpu_model_->outputs[out_index].shape;
  auto elem_size = 1;
  if (bpu_model_->outputs[out_index].data_type == BPU_TYPE_TENSOR_S32) {
    elem_size = 4;
  }
  auto shift = bpu_model_->outputs[out_index].shifts;
  float threshold_quanti = threshold * (1 << shift[0]);
  uint32_t src_n_stride =
      aligned_shape.d[1] * aligned_shape.d[2] * aligned_shape.d[3] * elem_size;
  uint32_t src_h_stride = aligned_shape.d[2] * aligned_shape.d[3] * elem_size;
  uint32_t src_w_stride = aligned_shape.d[3] * elem_size;

  float tmp_float_value;
  int32_t tmp_int32_value;

  int index = 0;
  valid_num = 0;
  for (int nn = 0; nn < real_shape.d[0]; nn++) {
    void *cur_n_src = reinterpret_cast<int8_t *>(src_ptr) + nn * src_n_stride;
    for (int hh = 0; hh < real_shape.d[1]; hh++) {
      void *cur_h_src =
          reinterpret_cast<int8_t *>(cur_n_src) + hh * src_h_stride;
      for (int ww = 0; ww < real_shape.d[2]; ww++) {
        void *cur_w_src =
            reinterpret_cast<int8_t *>(cur_h_src) + ww * src_w_stride;
        for (int cc = 0; cc < real_shape.d[3]; cc++) {
          void *cur_c_src =
              reinterpret_cast<int8_t *>(cur_w_src) + cc * elem_size;
          if (elem_size == 4) {
            tmp_int32_value = *(reinterpret_cast<int32_t *>(cur_c_src));
            if (tmp_int32_value >= threshold_quanti) {
              tmp_float_value = GetFloatByInt(tmp_int32_value, shift[cc]);
              *(reinterpret_cast<float *>(dest_ptr)) = tmp_float_value;
              dest_ptr = reinterpret_cast<char *>(dest_ptr) + elem_size;
              valid_index[valid_num] = index;
              valid_num++;
            }
          }
          index++;
        }
      }
    }
  }
}

void DetectPostProcessor::ConvertOutputFilter(
    void *src_ptr,
    void *dest_ptr,
    int out_index,
    const std::vector<int> &valid_index,
    const int &valid_num) {
  auto &aligned_shape = bpu_model_->outputs[out_index].aligned_shape;
  auto &real_shape = bpu_model_->outputs[out_index].shape;
  auto elem_size = 1;
  if (bpu_model_->outputs[out_index].data_type == BPU_TYPE_TENSOR_S32) {
    elem_size = 4;
  }
  auto shift = bpu_model_->outputs[out_index].shifts;

  uint32_t src_h_stride = aligned_shape.d[2] * aligned_shape.d[3];
  uint32_t src_w_stride = aligned_shape.d[3];

  float tmp_float_value;
  int32_t tmp_int32_value;

  for (int i = 0; i < valid_num; i++) {
    int index = valid_index[i];
    // 根据index计算h,w
    int h_index = index / real_shape.d[2];
    int w_index = index % real_shape.d[2];
    // 对有效索引处的数据定点转浮点
    for (int cc = 0; cc < real_shape.d[3]; cc++) {
      tmp_int32_value = *(reinterpret_cast<int32_t *>(
          src_ptr) + h_index * src_h_stride +
          w_index * src_w_stride + cc);
      tmp_float_value = GetFloatByInt(tmp_int32_value, shift[cc]);
      *(reinterpret_cast<float *>(dest_ptr)) = tmp_float_value;
      dest_ptr = reinterpret_cast<char *>(dest_ptr) + elem_size;
    }
  }
}

void DetectPostProcessor::ConvertOutputFilterFromFile(
    int i, void *dest_ptr,
    int out_index, float threshold,
    std::vector<int> &valid_index, int &valid_num) {
  const int stride_offset = feat_stride_.size() * 3;
  std::map<int, std::string> branch_name {
      {0, "head"},
      {1, "face"},
      {2, "hand"}
  };
  std::map<int, std::string> level_name {
      {0, "score"},
      {1, "regression"},
      {2, "centerness"}
  };
  std::string file_name = "./bins/" +
      branch_name[out_index / stride_offset] +
      "_branch_stride" + std::to_string(feat_stride_[i]) + "_" +
      level_name[out_index % stride_offset / feat_stride_.size()] + ".bin";

  float obj_f;
  std::fstream tri_f(file_name, std::ios::in | std::ios::binary);

  if (!tri_f) {
    LOGE << "open file: " << file_name << " failed";
    return;
  }
  int index = 0;
  while (tri_f.read(reinterpret_cast<char *>(&obj_f), sizeof(obj_f))) {
    valid_index[valid_num] = index;
    valid_num++;
    *(reinterpret_cast<float *>(dest_ptr)) = obj_f;
    dest_ptr = reinterpret_cast<char *>(dest_ptr) + sizeof(obj_f);
    index++;
  }
}

void DetectPostProcessor::ConvertOutputFromFile(
    int i, void *dest_ptr,
    int out_index,
    std::vector<int> &valid_index, const int &valid_num) {
  const int stride_offset = feat_stride_.size() * 3;
  std::map<int, std::string> branch_name {
      {0, "head"},
      {1, "face"},
      {2, "hand"}
  };
  std::map<int, std::string> level_name {
      {0, "score"},
      {1, "regression"},
      {2, "centerness"}
  };
  std::string file_name = "./bins/" +
      branch_name[out_index / stride_offset] +
      "_branch_stride" + std::to_string(feat_stride_[i]) + "_" +
      level_name[out_index % stride_offset / feat_stride_.size()] + ".bin";

  float obj_f;
  std::fstream tri_f(file_name, std::ios::in | std::ios::binary);

  if (!tri_f) {
    LOGE << "open file: " << file_name << "failed";
    return;
  }

  while (tri_f.read(reinterpret_cast<char *>(&obj_f), sizeof(obj_f))) {
    *(reinterpret_cast<float *>(dest_ptr)) = obj_f;
    dest_ptr = reinterpret_cast<char *>(dest_ptr) + sizeof(obj_f);
  }
}

}  // namespace xstream
