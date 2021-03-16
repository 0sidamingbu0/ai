/*
 * @Description: implement of vioplugin
 * @Author: zhuoran.rong@horizon.ai
 * @Date: 2020-04-08 16:17:25
 * @Copyright 2017~2019 Horizon Robotics, Inc.
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
#include "opencv2/opencv.hpp"
#include "utils/executor.h"

#include "vioplugin/viomessage.h"
#include "vioplugin/vioprocess.h"
#include "vioplugin/vioproduce.h"
#include "iotviomanager/viopipemanager.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

CachedImageList::CachedImageList(const char *vio_cfg_file) : VioProduce() {
#if defined(X3_X2_VIO)
  auto ret = hb_vio_init(vio_cfg_file);
  HOBOT_CHECK_EQ(ret, 0) << "vio init failed";
  ret = hb_vio_start();
  HOBOT_CHECK_EQ(ret, 0) << "vio start failed";
#elif defined(X3_IOT_VIO)
  std::vector<std::string> vio_cfg_list;
  vio_cfg_list.push_back(vio_cfg_file);
  VioPipeManager &manager = VioPipeManager::Get();
  if (-1 == pipe_id_) {
    pipe_id_ = manager.GetPipeId(vio_cfg_list);
  }
  HOBOT_CHECK(pipe_id_ != -1) << "CachedImageList: Get pipe_id failed";
  if (nullptr == vio_pipeline_) {
    vio_pipeline_ = std::make_shared<VioPipeLine>(vio_cfg_list, pipe_id_);
  }
  HOBOT_CHECK(vio_pipeline_ != nullptr)
      << "CachedImageList: Create VioPipeLine failed";
#endif
}

CachedImageList::~CachedImageList() {
#if defined(X3_X2_VIO)
  hb_vio_stop();
  hb_vio_deinit();
#elif defined(X3_IOT_VIO)
  if (vio_pipeline_) {
    vio_pipeline_ = nullptr;
  }
#endif
}

int CachedImageList::Run() {
  uint64_t frame_id = 0;
  // 每帧的时间间隔 ms
  int interval_ms = 500;

  bool enable_pyramid_profile = false;
  auto enable_pyramid_profile_str = getenv("enable_pyramid_profile");
  if (enable_pyramid_profile_str
      && !strcmp(enable_pyramid_profile_str, "ON")) {
    enable_pyramid_profile = true;
  }

  if (is_running_)
    return kHorizonVioErrorAlreadyStart;

  // 获得文件名数组
  auto json = config_->GetJson();
  // 图像文件列表
  auto cache_of_img_list = json["image_list"];

  if (cache_of_img_list.isNull()) {
    cache_of_img_list = Json::Value("");
  }

  if (cache_of_img_list.isString()) {
    auto file_list_obj = Json::Value();
    file_list_obj.resize(1);
    file_list_obj[0] = cache_of_img_list.asString();

    cache_of_img_list = file_list_obj;
  }

  auto interval_cfg = json["interval"];
  if (!interval_cfg.isNull()) {
    interval_ms = interval_cfg.asInt();
    HOBOT_CHECK(interval_ms >= 0) << "interval must great or equal than 0";
  }

  // 图像数据列表
  std::vector<HorizonVisionImage *> yuv420_data_cache;
  if (cam_type_ == "mono" || cam_type_ == "dual") {
    yuv420_data_cache.resize(cache_of_img_list.size());
    // 加载图像到内存
    for (unsigned int i = 0; i < cache_of_img_list.size(); ++i) {
      auto image = GetImageFrame(cache_of_img_list[i].asString());
      auto pad_width = std::stoi(config_->GetValue("pad_width"));
      auto pad_height = std::stoi(config_->GetValue("pad_height"));
      auto res = PadImage(&image->image, pad_width, pad_height);
      HOBOT_CHECK(res == 0)
          << "Failed to pad image " << cache_of_img_list[i].asString();
      HorizonVisionImage *nv12;
      HorizonVisionAllocImage(&nv12);
      HorizonConvertImage(&image->image, nv12,
                          kHorizonVisionPixelFormatRawNV12);
      HorizonVisionFreeImageFrame(image);
      yuv420_data_cache[i] = nv12;
    }
  } else {
    LOGE << "Unsupport camera type : " << cam_type_;
    return -1;
  }
  LOGI << "Finish importing images";

  is_running_ = true;

  while (1) {
    // 循环这些list, 循环读出一个 sid => source id
    for (unsigned int sid = 0; sid < cache_of_img_list.size(); sid++) {
      if (!is_running_) {
        goto exit_loop;
      }
      if (cam_type_ == "mono") {
        if (AllocBuffer()) {
          auto pym_start_time = std::chrono::system_clock::now();
#if defined(X3_X2_VIO) || defined(X3_IOT_VIO)
          VioFeedbackContext *feedback_context =
              reinterpret_cast<VioFeedbackContext *>(
                  std::calloc(1, sizeof(VioFeedbackContext)));
            char *yuv_data =
                reinterpret_cast<char *>(yuv420_data_cache[sid]->data);
            int yuv_width = yuv420_data_cache[sid]->width;
            int yuv_height = yuv420_data_cache[sid]->height;
            bool ret = GetPyramidInfo(vio_pipeline_, feedback_context, yuv_data,
                                      yuv_width * yuv_height * 3 / 2);
#endif

            std::vector<std::shared_ptr<PymImageFrame>> pym_images;
            if (ret) {
              auto pym_image_frame_ptr = std::make_shared<PymImageFrame>();
              Convert(&feedback_context->pym_img_info, *pym_image_frame_ptr);
              pym_image_frame_ptr->channel_id = sid;
              pym_image_frame_ptr->frame_id = frame_id;
              pym_image_frame_ptr->time_stamp = frame_id++;
              // set context to feedback_context
              pym_image_frame_ptr->context =
                  static_cast<void *>(feedback_context);
              pym_images.push_back(pym_image_frame_ptr);
            } else {
              std::free(feedback_context);
              LOGF << "GetPyramidInfo failed, ret: " << ret;
              continue;
              // HOBOT_CHECK(ret == true);
            }
            std::shared_ptr<VioMessage> input(
                new ImageVioMessage(vio_pipeline_, pym_images, 1),
                [&](ImageVioMessage *p) {
                  if (p) {
#if defined(X3_X2_VIO) || defined(X3_IOT_VIO)
                    p->FreeImage(1);
#endif
                    FreeBuffer();
                    delete p;
                  }
                  p = nullptr;
                });
            auto pym_end_time = std::chrono::system_clock::now();
            if (enable_pyramid_profile) {
              auto duration_time =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  pym_end_time - pym_start_time);
              LOGW << "pym time: " << duration_time.count() << " ms";
            }
            if (push_data_cb_)
              push_data_cb_(input);
            LOGD << "Push Image message!!!";
          } else {
            // 产生Drop帧
            LOGV << "NO VIO BUFFER";
            auto input = std::make_shared<DropVioMessage>(frame_id, frame_id++);
            if (push_data_cb_)
              push_data_cb_(input);
            LOGI << "Push Drop message!!!";
          }
        } else if (cam_type_ == "dual") {
          // todo
          LOGF << "do not support dual yet";
        }
        // 暂停
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
      }
    }

  exit_loop:
    // 释放资源
    if (cam_type_ == "mono") {
      for (unsigned int i = 0; i < yuv420_data_cache.size(); ++i) {
        HorizonVisionFreeImage(yuv420_data_cache[i]);
      }
      yuv420_data_cache.clear();
    } else if (cam_type_ == "dual") {
    }

    return 0;
  }  // namespace vioplugin

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
