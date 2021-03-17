/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-11-12
 * @Version: v0.0.1
 * @Brief: implement of video feedback produce.
 */
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hobotlog/hobotlog.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "utils/executor.h"

#include "hobotxstream/image_tools.h"
#include "vioplugin/viomessage.h"
#include "vioplugin/vioprocess.h"
#include "vioplugin/vioproduce.h"
#include "iotviomanager/viopipemanager.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

MultiFeedbackProduce::MultiFeedbackProduce(
    const std::vector<std::string> &vio_cfg_list) {
  HOBOT_CHECK(vio_cfg_list.size() > 0) << "vio cfg file is null";
#ifdef X3_IOT_VIO
  VioPipeManager &manager = VioPipeManager::Get();
  if (-1 == pipe_id_) {
    pipe_id_ = manager.GetPipeId(vio_cfg_list);
  }
  HOBOT_CHECK(pipe_id_ != -1) << "MultiFeedbackProduce: Get pipe_id failed";
  if (nullptr == vio_pipeline_) {
    LOGI << "ApaCameara create viopipeline pipe_id: " << pipe_id_;
    vio_pipeline_ = std::make_shared<VioPipeLine>(vio_cfg_list, pipe_id_);
  }
  HOBOT_CHECK(vio_pipeline_ != nullptr)
    << "MultiFeedbackProduce: Create VioPipeLine failed";
#endif
}

MultiFeedbackProduce::~MultiFeedbackProduce() {
#ifdef X3_IOT_VIO
  if (vio_pipeline_) {
    vio_pipeline_ = nullptr;
  }
#endif
}

bool MultiFeedbackProduce::FillVIOImageByImagePath(
    const std::vector<std::string> &image_name_list,
    std::vector<std::shared_ptr<PymImageFrame>> &pym_images) {
  int len, y_img_len, uv_img_len;
  char *data = nullptr;
  auto img_num = image_name_list.size();

  /* 1. get multi fb src info */
  std::vector<std::shared_ptr<SrcImageFrame>> src_images;
  auto ret = vio_pipeline_->GetMultiFbSrcInfo(src_images);
  if (ret < 0) {
    LOGE << "vio pipeline get multi feedback src info failed";
    return false;
  }
  HOBOT_CHECK(img_num == src_images.size());

  /* 2. copy image data to fb src addr */
  for (uint32_t index = 0; index < img_num; index++) {
    auto img_name = image_name_list[index];
    if (fb_mode_ == "jpeg_image_list") {
      auto image = GetImageFrame(img_name);
      auto pad_width = std::stoi(config_->GetValue("pad_width"));
      auto pad_height = std::stoi(config_->GetValue("pad_height"));
      auto res = PadImage(&image->image, pad_width, pad_height);
      if (res != 0) {
        LOGE << "Failed to pad image " << img_name << ", error code is " << res;
        return false;
      }
      HorizonVisionImage *nv12;
      HorizonVisionAllocImage(&nv12);
      HorizonConvertImage(&image->image, nv12,
          kHorizonVisionPixelFormatRawNV12);
      data = reinterpret_cast<char *>(nv12->data);
      len = nv12->height * nv12->width * 3 / 2;
      y_img_len = len / 3 * 2;
      uv_img_len = len / 3;
      memcpy(reinterpret_cast<uint8_t *>(src_images[index]->src_info.y_vaddr),
          data, y_img_len);
      memcpy(reinterpret_cast<uint8_t *>(src_images[index]->src_info.c_vaddr),
          data + y_img_len, uv_img_len);
      HorizonVisionFreeImage(nv12);
      HorizonVisionFreeImageFrame(image);
    } else if (fb_mode_ == "nv12_image_list") {
      if (access(img_name.c_str(), F_OK) != 0) {
        LOGE << "File not exist: " << img_name;
        return false;
      }
      std::ifstream ifs(img_name, std::ios::in | std::ios::binary);
      if (!ifs) {
        LOGE << "Failed load " << img_name;
      }
      ifs.seekg(0, std::ios::end);
      int len = ifs.tellg();
      ifs.seekg(0, std::ios::beg);
      data = new char[len];
      ifs.read(data, len);
      y_img_len = len / 3 * 2;
      uv_img_len = len / 3;
      memcpy(reinterpret_cast<uint8_t *>(src_images[index]->src_info.y_vaddr),
          data, y_img_len);
      memcpy(reinterpret_cast<uint8_t *>(src_images[index]->src_info.c_vaddr),
          data + y_img_len, uv_img_len);
      delete[] data;
      ifs.close();
    } else {
      LOGF << "Don't support fb mode: " << fb_mode_;
      return false;
    }
  }

  /* 3. send multi image data to pym */
  ret = vio_pipeline_->SetMultiFbPymInfo(src_images);
  if (ret < 0) {
    LOGE << "vio pipeline set multi feedback pym process info failed";
    return false;
  }

  /* 4. get multi pym result */
  ret = vio_pipeline_->GetMultiPymInfo(pym_images);
  if (ret < 0) {
    LOGE << "vio pipeline get multi pym info failed";
    return false;
  }

  return true;
}

int MultiFeedbackProduce::ParseImageListFile() {
  std::string image_path;
  std::vector<std::string> image_list_file;
  uint32_t all_img_count = 0;

  auto json = config_->GetJson();
  name_list_loop_ = json["name_list_loop"].asInt();
  fb_mode_ = json["multi_feedback_mode"].asString();
  auto interval_cfg = json["interval"];
  if (!interval_cfg.isNull()) {
    interval_ms_ = interval_cfg.asInt();
    HOBOT_CHECK(interval_ms_ >= 0) << "interval must great or equal than 0";
  }

  auto list_of_img_list = json["file_path"];
  if (list_of_img_list.isNull()) {
    list_of_img_list = Json::Value("");
  }

  if (list_of_img_list.isString()) {
    auto file_list_obj = Json::Value();
    file_list_obj.resize(1);
    file_list_obj[0] = list_of_img_list.asString();

    list_of_img_list = file_list_obj;
  }

  for (unsigned int i = 0; i < list_of_img_list.size(); i++) {
    image_list_file.push_back(list_of_img_list[i].asString());
  }

  image_source_list_.resize(image_list_file.size());
  uint32_t s_img_cnt_tmp = 0;
  for (unsigned int i = 0; i < image_list_file.size(); ++i) {
    std::ifstream ifs(image_list_file[i]);

    if (!ifs.good()) {
      LOGF << "Open file failed: " << image_list_file[i];
      return -1;
    }


    while (std::getline(ifs, image_path)) {
      // trim the spaces in the beginning
      image_path.erase(0, image_path.find_first_not_of(' '));
      // trim the spaces in the end
      image_path.erase(image_path.find_last_not_of(' ') + 1, std::string::npos);

      image_source_list_[i].emplace_back(image_path);
    }
    //记录单路图片总数
    s_img_cnt_tmp = image_source_list_[i].size();
    if (s_img_cnt_tmp > s_img_cnt_) {
      // 获取最大图片数的那一路
      s_img_cnt_ = s_img_cnt_tmp;
    }

    // 记录图片总数
    all_img_count += s_img_cnt_tmp;
    ifs.close();
  }

  pipe_num_ = image_list_file.size();
  auto all_img_cnt_tmp = s_img_cnt_ * pipe_num_;
  HOBOT_CHECK(all_img_cnt_tmp == all_img_count);
  LOGD << "Finish importing images";

#if 0
  uint32_t img_num = 0;
  std::vector<std::string> image_name_list;
  LOGD << "create feed image task ";
  sp_feed_image_task_ = std::make_shared<std::thread>(
      [&] () {
      while (is_running_) {
        image_name_list.clear();
        if (img_num >= s_img_cnt_) {
          if (name_list_loop_) {
            img_num = 0;
          } else {
            is_running_ = false;
          }
        continue;
      }
      for (uint32_t pipe_idx = 0; pipe_idx < pipe_num_; ++pipe_idx) {
        auto image_name = image_source_list_[pipe_idx][img_num];
        image_name_list.push_back(image_name);
      }
      img_num++;
      FillVIOImageByImagePath(image_name_list);
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
      }
      });
#endif

  return 0;
}

int MultiFeedbackProduce::Run() {
  uint64_t frame_id = 0;
  uint32_t img_num = 0;
  std::vector<std::string> image_name_list;

  auto ret = ParseImageListFile();
  HOBOT_CHECK(ret == 0) << "parse image list file error!!!";
  if (is_running_)
    return kHorizonVioErrorAlreadyStart;
  is_running_ = true;

  while (is_running_) {
    image_name_list.clear();
    if (img_num >= s_img_cnt_) {
      if (name_list_loop_) {
        img_num = 0;
      } else {
        is_running_ = false;
      }
      continue;
    }
    // 分配Buffer. 等待Buffer可用
    while (!AllocBuffer()) {
      LOGW << "NO VIO_FB_BUFFER";
      std::this_thread::sleep_for(std::chrono::microseconds(5));
      continue;
    }
    for (uint32_t pipe_idx = 0; pipe_idx < pipe_num_; ++pipe_idx) {
      auto image_name = image_source_list_[pipe_idx][img_num];
      image_name_list.push_back(image_name);
    }
    img_num++;

    std::vector<std::shared_ptr<PymImageFrame>> pym_images;
    ret = FillVIOImageByImagePath(image_name_list, pym_images);
    if (ret) {
      for (uint32_t index = 0; index < pym_images.size(); index++) {
        pym_images[index]->frame_id = frame_id;
        pym_images[index]->time_stamp = frame_id;
        LOGD << "vio channel_id: " << pym_images[index]->channel_id
          << " timestamp: " << pym_images[index]->time_stamp;
      }
      frame_id++;
    } else {
      LOGF << "fill vio image failed, ret: " << ret;
      continue;
    }

    std::shared_ptr<VioMessage> input(
        new ImageVioMessage(vio_pipeline_, pym_images, pym_images.size()),
        [&](ImageVioMessage *p) {
        if (p) {
        LOGD << "begin delete ImageVioMessage";
        p->FreeMultiImage();
        FreeBuffer();
        delete (p);
        }
        p = nullptr;
        });
    if (push_data_cb_) {
      push_data_cb_(input);
      LOGD << "Push Image message!!!";
    }
     // interval ms
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
  }  // while is_running_ loop
  is_running_ = false;
  return 0;
}

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
