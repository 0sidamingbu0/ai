/**
 * Copyright (c) 2019, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: Songshan Gong
 * @Mail: songshan.gong@horizon.ai
 * @Date: 2019-08-01 20:38:52
 * @Version: v0.0.1
 * @Brief: smartplugin impl based on xstream.
 * @Last Modified by: Songshan Gong
 * @Last Modified time: 2019-09-29 05:04:11
 */
#include "rtspplugin/rtspplugin.h"

#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "BasicUsageEnvironment.hh"
#include "hobotlog/hobotlog.hpp"
#include "hobotxsdk/xstream_sdk.h"
#include "horizon/vision/util.h"
#include "horizon/vision_type/vision_error.h"
#include "horizon/vision_type/vision_msg.h"
#include "horizon/vision_type/vision_type_util.h"
#include "liveMedia.hh"
#include "mediapipemanager/mediapipemanager.h"
#include "mediapipemanager/meidapipeline.h"
#include "rtspclient/SPSInfoMgr.h"
#include "rtspplugin/rtspmessage.h"
#include "smartplugin_box/displayinfo.h"
#include "unistd.h"
#include "video_box_common.h"
#include "videoprocessor.h"
#include "xproto/message/pluginflow/flowmsg.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto/plugin/xpluginasync.h"
#include "xproto_msgtype/vioplugin_data.h"
// #include "mediapipemanager/meidapipelinetest.h"

// #define PIPE_TEST

namespace horizon {
namespace vision {
namespace xproto {
namespace rtspplugin {

using horizon::vision::xproto::XPluginAsync;
using horizon::vision::xproto::XProtoMessage;
using horizon::vision::xproto::XProtoMessagePtr;
using horizon::vision::xproto::basic_msgtype::VioMessage;

using hobot::vision::PymImageFrame;

// using horizon::vision::xproto::basic_msgtype::VioMessage;
using ImageFramePtr = std::shared_ptr<hobot::vision::ImageFrame>;
using XStreamImageFramePtr = xstream::XStreamData<ImageFramePtr>;

using xstream::InputDataPtr;
using xstream::OutputDataPtr;
using xstream::XStreamSDK;

XPLUGIN_REGISTER_MSG_TYPE(TYPE_DECODE_IMAGE_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(TYPE_DECODE_DROP_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_IMAGE_MESSAGE)
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_DROP_MESSAGE)

int RtspPlugin::frame_count_ = 1;
std::mutex RtspPlugin::framecnt_mtx_;

int AllocPymBuffer(pym_buffer_t *pym_img, const bool alloc_720P) {
  int width = 1920;
  int height = 1080;
  pym_img->pym[0].width = width;
  pym_img->pym[0].height = height;
  pym_img->pym[0].stride_size = width;

  pym_img->pym[1].width = width / 2;
  pym_img->pym[1].height = height / 2;
  pym_img->pym[1].stride_size = width / 2;

  if (alloc_720P) {
    pym_img->pym_roi[0][0].width = 1280;
    pym_img->pym_roi[0][0].height = 720;
    pym_img->pym_roi[0][0].stride_size = 720;
  }

  pym_img->pym_roi[1][0].width = 640;
  pym_img->pym_roi[1][0].height = 360;
  pym_img->pym_roi[1][0].stride_size = 360;

  int nRet = HB_SYS_Alloc(&pym_img->pym[0].paddr[0],
                          reinterpret_cast<void **>(&pym_img->pym[0].addr[0]),
                          1920 * 1080);
  if (nRet) {
    LOGE << "video box rtspplugin HB_SYS_Alloc fail";
    return nRet;
  } else {
    printf("mmzAlloc pym 1080p paddr0 = 0x%lx, vaddr = 0x%p \n",
           pym_img->pym[0].paddr[0], &pym_img->pym[0].addr[0]);
  }
  nRet = HB_SYS_Alloc(&pym_img->pym[0].paddr[1],
                      reinterpret_cast<void **>(&pym_img->pym[0].addr[1]),
                      1920 * 1080 / 2);
  if (nRet) {
    LOGE << "video box rtspplugin HB_SYS_Alloc fail";
    return nRet;
  }
  nRet = HB_SYS_Alloc(&pym_img->pym[1].paddr[0],
                      reinterpret_cast<void **>(&pym_img->pym[1].addr[0]),
                      960 * 540);
  if (nRet) {
    LOGE << "video box rtspplugin HB_SYS_Alloc fail";
    return nRet;
  }
  nRet = HB_SYS_Alloc(&pym_img->pym[1].paddr[1],
                      reinterpret_cast<void **>(&pym_img->pym[1].addr[1]),
                      960 * 540 / 2);
  if (nRet) {
    LOGE << "video box rtspplugin HB_SYS_Alloc fail";
    return nRet;
  }

  if (alloc_720P) {
    nRet = HB_SYS_Alloc(
        &pym_img->pym_roi[0][0].paddr[0],
        reinterpret_cast<void **>(&pym_img->pym_roi[0][0].addr[0]), 1280 * 720);
    if (nRet) {
      LOGE << "video box rtspplugin HB_SYS_Alloc fail";
      return nRet;
    }

    nRet =
        HB_SYS_Alloc(&pym_img->pym_roi[0][0].paddr[1],
                     reinterpret_cast<void **>(&pym_img->pym_roi[0][0].addr[1]),
                     1280 * 720 / 2);
    if (nRet) {
      LOGE << "video box rtspplugin HB_SYS_Alloc fail";
      return nRet;
    }
  }

  nRet = HB_SYS_Alloc(
      &pym_img->pym_roi[1][0].paddr[0],
      reinterpret_cast<void **>(&pym_img->pym_roi[1][0].addr[0]), 640 * 360);
  if (nRet) {
    LOGE << "video box rtspplugin HB_SYS_Alloc fail";
    return nRet;
  }

  nRet =
      HB_SYS_Alloc(&pym_img->pym_roi[1][0].paddr[1],
                   reinterpret_cast<void **>(&pym_img->pym_roi[1][0].addr[1]),
                   640 * 360 / 2);
  if (nRet) {
    LOGE << "video box rtspplugin HB_SYS_Alloc fail";
    return nRet;
  }
  return 0;
}

void FreePymBuffer(pym_buffer_t *pym_img, const bool alloc_720P) {
  HB_SYS_Free(pym_img->pym[0].paddr[0], pym_img->pym[0].addr[0]);
  HB_SYS_Free(pym_img->pym[0].paddr[1], pym_img->pym[0].addr[1]);
  HB_SYS_Free(pym_img->pym[1].paddr[0], pym_img->pym[1].addr[0]);
  HB_SYS_Free(pym_img->pym[1].paddr[1], pym_img->pym[1].addr[1]);
  if (alloc_720P) {
    HB_SYS_Free(pym_img->pym_roi[0][0].paddr[0],
                pym_img->pym_roi[0][0].addr[0]);
    HB_SYS_Free(pym_img->pym_roi[0][0].paddr[1],
                pym_img->pym_roi[0][0].addr[1]);
  }
  HB_SYS_Free(pym_img->pym_roi[1][0].paddr[0], pym_img->pym_roi[1][0].addr[0]);
  HB_SYS_Free(pym_img->pym_roi[1][0].paddr[1], pym_img->pym_roi[1][0].addr[1]);
}

void Convert(pym_buffer_t *pym_buffer, hobot::vision::PymImageFrame &pym_img) {
  if (nullptr == pym_buffer) {
    return;
  }
  pym_img.ds_pym_total_layer = DOWN_SCALE_MAX;
  pym_img.us_pym_total_layer = UP_SCALE_MAX;
  pym_img.frame_id = pym_buffer->pym_img_info.frame_id;
  pym_img.time_stamp = pym_buffer->pym_img_info.time_stamp;
  pym_img.context = static_cast<void *>(pym_buffer);
  for (int i = 0; i < DOWN_SCALE_MAX; ++i) {
    address_info_t *pym_addr = NULL;
    if (i % 4 == 0) {
      pym_addr = reinterpret_cast<address_info_t *>(&pym_buffer->pym[i / 4]);
    } else {
      pym_addr = reinterpret_cast<address_info_t *>(
          &pym_buffer->pym_roi[i / 4][i % 4 - 1]);
    }
    // std::cout << "dxd1 : " << pym_addr->width << std::endl;
    pym_img.down_scale[i].width = pym_addr->width;
    pym_img.down_scale[i].height = pym_addr->height;
    pym_img.down_scale[i].stride = pym_addr->stride_size;
    pym_img.down_scale[i].y_paddr = pym_addr->paddr[0];
    pym_img.down_scale[i].c_paddr = pym_addr->paddr[1];
    pym_img.down_scale[i].y_vaddr =
        reinterpret_cast<uint64_t>(pym_addr->addr[0]);
    pym_img.down_scale[i].c_vaddr =
        reinterpret_cast<uint64_t>(pym_addr->addr[1]);
  }
  for (int i = 0; i < UP_SCALE_MAX; ++i) {
    pym_img.up_scale[i].width = pym_buffer->us[i].width;
    pym_img.up_scale[i].height = pym_buffer->us[i].height;
    pym_img.up_scale[i].stride = pym_buffer->us[i].stride_size;
    pym_img.up_scale[i].y_paddr = pym_buffer->us[i].paddr[0];
    pym_img.up_scale[i].c_paddr = pym_buffer->us[i].paddr[1];
    pym_img.up_scale[i].y_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->us[i].addr[0]);
    pym_img.up_scale[i].c_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->us[i].addr[1]);
  }
  for (int i = 0; i < DOWN_SCALE_MAIN_MAX; ++i) {
    // std::cout << "dxd2 : " << pym_buffer->pym[i].width << std::endl;
    pym_img.down_scale_main[i].width = pym_buffer->pym[i].width;
    pym_img.down_scale_main[i].height = pym_buffer->pym[i].height;
    pym_img.down_scale_main[i].stride = pym_buffer->pym[i].stride_size;
    pym_img.down_scale_main[i].y_paddr = pym_buffer->pym[i].paddr[0];
    pym_img.down_scale_main[i].c_paddr = pym_buffer->pym[i].paddr[1];
    pym_img.down_scale_main[i].y_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->pym[i].addr[0]);
    pym_img.down_scale_main[i].c_vaddr =
        reinterpret_cast<uint64_t>(pym_buffer->pym[i].addr[1]);
  }
}

void Convert_ex(pym_buffer_t *pym_buffer, pym_buffer_t &pym_img,
                const bool convert_720P, const int channel) {
  for (int i = 0; i < 2; ++i) {
    address_info_t *pym_addr = NULL;
    pym_addr = reinterpret_cast<address_info_t *>(&pym_buffer->pym[i]);
    pym_img.pym[i].width = pym_addr->width;
    pym_img.pym[i].height = pym_addr->height;
    pym_img.pym[i].stride_size = pym_addr->stride_size;
    int width_tmp = pym_addr->width;
    int height_tmp = pym_addr->height;
    int size = width_tmp * height_tmp;
    if (pym_addr->width == pym_addr->stride_size) {
      memcpy(pym_img.pym[i].addr[0], pym_addr->addr[0], size);
      memcpy(pym_img.pym[i].addr[1], pym_addr->addr[1], size / 2);
    } else {
      // jump over stride - width Y
      for (i = 0; i < height_tmp; i++) {
        memcpy(pym_img.pym[i].addr[0] + i * width_tmp,
               pym_addr->addr[0] + i * pym_addr->stride_size, width_tmp);
      }

      // jump over stride - width UV
      for (i = 0; i < height_tmp / 2; i++) {
        memcpy(pym_img.pym[i].addr[1] + i * width_tmp,
               pym_addr->addr[1] + i * pym_addr->stride_size, width_tmp);
      }
    }
  }  // end of for

  if (convert_720P) {
    pym_img.pym_roi[0][0].width = pym_buffer->pym_roi[0][0].width;
    pym_img.pym_roi[0][0].height = pym_buffer->pym_roi[0][0].height;
    pym_img.pym_roi[0][0].stride_size = pym_buffer->pym_roi[0][0].stride_size;
    memcpy(pym_img.pym_roi[0][0].addr[0], pym_buffer->pym_roi[0][0].addr[0],
           pym_buffer->pym_roi[0][0].width * pym_buffer->pym_roi[0][0].height);
    memcpy(
        pym_img.pym_roi[0][0].addr[1], pym_buffer->pym_roi[0][0].addr[1],
        pym_buffer->pym_roi[0][0].width * pym_buffer->pym_roi[0][0].height / 2);
  }

  pym_img.pym_roi[1][0].width = pym_buffer->pym_roi[1][0].width;
  pym_img.pym_roi[1][0].height = pym_buffer->pym_roi[1][0].height;
  pym_img.pym_roi[1][0].stride_size = pym_buffer->pym_roi[1][0].stride_size;
  memcpy(pym_img.pym_roi[1][0].addr[0], pym_buffer->pym_roi[1][0].addr[0],
         pym_buffer->pym_roi[1][0].width * pym_buffer->pym_roi[1][0].height);
  memcpy(
      pym_img.pym_roi[1][0].addr[1], pym_buffer->pym_roi[1][0].addr[1],
      pym_buffer->pym_roi[1][0].width * pym_buffer->pym_roi[1][0].height / 2);
  pym_img.pym_img_info.pipeline_id = channel;
}

char eventLoopWatchVariable = 0;

int RtspPlugin::Init() {
  running_ = false;
  GetConfigFromFile(config_file_);
  SPSInfoMgr::GetInstance().Init();
  MediaPipeManager::GetInstance().Init();
  for (int i = 0; i < channel_number_; ++i) {
    std::shared_ptr<horizon::vision::MediaPipeLine> pipeline =
        std::make_shared<horizon::vision::MediaPipeLine>(i, i);
    pipeline->SetFrameDropFlag(drop_frame_);
    pipeline->SetFrameDropInterval(drop_frame_interval_);
    if (draw_real_time_video_) {
      pipeline->SetVpsFrameDepth(2);
    }
    MediaPipeManager::GetInstance().AddPipeLine(pipeline);
  }

  return XPluginAsync::Init();
}

void RtspPlugin::ProcessData(const int channel_id, pym_buffer_t *pym_buffer) {
  int layer =
      DisplayInfo::computePymLayer(display_mode_, channel_number_, channel_id);
  address_info_t *pym_addr = NULL;
  if (layer % 4 == 0) {
    pym_addr = reinterpret_cast<address_info_t *>(&pym_buffer->pym[layer / 4]);
  } else {
    pym_addr = reinterpret_cast<address_info_t *>(
        &pym_buffer->pym_roi[layer / 4][layer % 4 - 1]);
  }
  // std::cout << "dowm scale layer:" << layer << " width:" << pym_addr->width
  //           << " height:" << pym_addr->height << std::endl;
  auto video_data = std::make_shared<horizon::vision::VideoData>();
  video_data->smart_frame = nullptr;
  uint32_t width_tmp = pym_addr->width;
  uint32_t height_tmp = pym_addr->height;
  char *y_addr = pym_addr->addr[0];
  char *uv_addr = pym_addr->addr[1];
  video_data->channel = channel_id;
  video_data->width = width_tmp;
  video_data->height = height_tmp;
  video_data->data_len = width_tmp * height_tmp * 3 / 2;
  video_data->buffer =
      static_cast<char *>(malloc(width_tmp * height_tmp * 3 / 2));
  for (uint32_t i = 0; i < height_tmp; ++i) {
    memcpy(video_data->buffer + i * width_tmp, y_addr + i * width_tmp,
           width_tmp);
  }

  for (uint32_t i = 0; i < (height_tmp / 2); ++i) {
    memcpy(video_data->buffer + (i + height_tmp) * width_tmp,
           uv_addr + i * width_tmp, width_tmp);
  }

  VideoProcessor::GetInstance().Input(video_data);

  if (running_venc_720p_ && !encode_smart_) {
    layer = DisplayInfo::computePymLayer(display_mode_, channel_number_,
                                         channel_id);
    if (layer % 4 == 0) {
      pym_addr =
          reinterpret_cast<address_info_t *>(&pym_buffer->pym[layer / 4]);
    } else {
      pym_addr = reinterpret_cast<address_info_t *>(
          &pym_buffer->pym_roi[layer / 4][layer % 4 - 1]);
    }
    auto video_data = std::make_shared<horizon::vision::VideoData>();
    video_data->smart_frame = nullptr;
    uint32_t width_tmp = pym_addr->width;
    uint32_t height_tmp = pym_addr->height;
    char *y_addr = pym_addr->addr[0];
    char *uv_addr = pym_addr->addr[1];
    video_data->channel = channel_id;
    video_data->width = width_tmp;
    video_data->height = height_tmp;
    video_data->data_len = width_tmp * height_tmp * 3 / 2;
    video_data->buffer =
        static_cast<char *>(malloc(width_tmp * height_tmp * 3 / 2));
    for (uint32_t i = 0; i < height_tmp; ++i) {
      memcpy(video_data->buffer + i * width_tmp, y_addr + i * width_tmp,
             width_tmp);
    }

    for (uint32_t i = 0; i < (height_tmp / 2); ++i) {
      memcpy(video_data->buffer + (i + height_tmp) * width_tmp,
             uv_addr + i * width_tmp, width_tmp);
    }

    VideoProcessor::GetInstance().Input(video_data, true);
  }

  if (transition_support_) {
    layer = 0;
    if (layer % 4 == 0) {
      pym_addr =
          reinterpret_cast<address_info_t *>(&pym_buffer->pym[layer / 4]);
    } else {
      pym_addr = reinterpret_cast<address_info_t *>(
          &pym_buffer->pym_roi[layer / 4][layer % 4 - 1]);
    }
    auto video_data = std::make_shared<horizon::vision::VideoData>();
    video_data->smart_frame = nullptr;
    uint32_t width_tmp = pym_addr->width;
    uint32_t height_tmp = pym_addr->height;
    char *y_addr = pym_addr->addr[0];
    char *uv_addr = pym_addr->addr[1];
    video_data->channel = channel_id;
    video_data->width = width_tmp;
    video_data->height = height_tmp;
    video_data->data_len = width_tmp * height_tmp * 3 / 2;
    video_data->buffer =
        static_cast<char *>(malloc(width_tmp * height_tmp * 3 / 2));
    for (uint32_t i = 0; i < height_tmp; ++i) {
      memcpy(video_data->buffer + i * width_tmp, y_addr + i * width_tmp,
             width_tmp);
    }

    for (uint32_t i = 0; i < (height_tmp / 2); ++i) {
      memcpy(video_data->buffer + (i + height_tmp) * width_tmp,
             uv_addr + i * width_tmp, width_tmp);
    }
    VideoProcessor::GetInstance().Input(video_data, false, true);
  }
}

void RtspPlugin::GetDeocdeFrame(std::shared_ptr<MediaPipeLine> pipeline,
                                int channel) {
  pym_buffer_t *out_pym_buf = nullptr;
  int ret = 0;

  while (running_) {
    ret = pipeline->Output((void **)(&out_pym_buf));
    if (ret != 0) {
      if (ret == -5) {  // not ready
        usleep(500 * 200);
        continue;
      }
      LOGI << "channel:" << channel << " Frame Drop";
      continue;
    }
    if (out_pym_buf == NULL) {
      LOGE << "mediapipeline output null pym buf, but not return error!";
      continue;
    }

    if (not_run_smart_) {
      ProcessData(channel, out_pym_buf);
      pipeline->OutputBufferFree(out_pym_buf);
      continue;
    }

    std::vector<std::shared_ptr<PymImageFrame>> pym_images;
    auto pym_image_frame_ptr = std::make_shared<PymImageFrame>();
    if (pym_image_frame_ptr == NULL) {
      LOGE << "make shared ptr fail, return null pointer!";
      continue;
    }
    int index = -1;
    if (draw_real_time_video_) {
      ProcessData(channel, out_pym_buf);
      pym_buffer_t *pym_image = nullptr;
      {
        std::lock_guard<std::mutex> lg(pym_buffer_mutex_);
        for (size_t i = 0; i < use_flag_list.size(); ++i) {
          if (use_flag_list[i]) {
            continue;
          }
          index = i;
          break;
        }
        if (index == -1) {
          LOGI << "rtspplugin has no pym buffer, drop frame:" << frame_count_++;
          pipeline->OutputBufferFree(out_pym_buf);
          continue;
        }
        use_flag_list[index] = true;
        pym_image = pym_buffers_[index];
      }
      Convert_ex(out_pym_buf, *pym_image, (running_venc_720p_ && encode_smart_),
                 channel);
      pipeline->OutputBufferFree(out_pym_buf);
      Convert(pym_image, *pym_image_frame_ptr);
    } else {
      Convert(out_pym_buf, *pym_image_frame_ptr);
    }

    pym_image_frame_ptr->channel_id = channel;
    {
      std::lock_guard<std::mutex> lg(framecnt_mtx_);
      pym_image_frame_ptr->frame_id = frame_count_++;
    }
    pym_images.push_back(pym_image_frame_ptr);

    auto free_buff = [&](ImageVioMessage *p) {
      if (p) {
        if (!draw_real_time_video_) {
          if (p->pipeline_ != nullptr) {
            p->pipeline_->OutputBufferFree(p->slot_data_);
          }
        } else {
          std::lock_guard<std::mutex> lg(pym_buffer_mutex_);
          use_flag_list[p->index_] = false;
        }
        delete p;
      }
      p = nullptr;
    };

    std::shared_ptr<VioMessage> input(
        new ImageVioMessage(pym_images, 1, 1, channel, pipeline,
                            out_pym_buf, index), free_buff);
    LOGD << "channel:" << channel
         << "image vio message construct  grp:" << pipeline->GetGrpId()
         << "  frame_id:" << pym_image_frame_ptr->frame_id;
    PushMsg(input);
  }
}

void RtspPlugin::Process() {
  // Begin by setting up our usage environment:
  scheduler_ = BasicTaskScheduler::createNew();
  env_ = BasicUsageEnvironment::createNew(*scheduler_);

  for (int i = 0; i < channel_number_; ++i) {
    ourRTSPClient *client = nullptr;
    client = openURL(*env_, "RTSPClient", rtsp_url_[i].url.c_str(),
                     rtsp_url_[i].tcp_flag, rtsp_url_[i].frame_max_size,
                     ("channel" + std::to_string(i) + ".stream"),
                     rtsp_url_[i].save_stream, i);
    rtsp_clients_.push_back(client);
  }

  const std::vector<std::shared_ptr<MediaPipeLine>> &pipelines =
      MediaPipeManager::GetInstance().GetPipeLine();
  LOGE << "\n\nPipeline size : " << pipelines.size();
  running_ = true;
  std::thread *t[channel_number_];
  for (uint32_t i = 0; i < pipelines.size(); ++i) {
    t[i] = new std::thread(&RtspPlugin::GetDeocdeFrame, this, pipelines[i], i);
  }

  // All subsequent activity takes place within the event loop:
  env_->taskScheduler().doEventLoop(&eventLoopWatchVariable);
  for (int i = 0; i < channel_number_; ++i) {
    ourRTSPClient *client = rtsp_clients_[i];
    // operators cause crash if client is invalid
    if (rtsp_clients_stat_.at(i)) {
      client->sendTeardownCommand(*client->scs.session, NULL);
      Medium::close(client->scs.session);
    }
  }

  env_->reclaim();
  delete scheduler_;
  for (uint32_t i = 0; i < pipelines.size(); ++i) {
    if (t[i]) {
      if (t[i]->joinable()) {
        t[i]->join();
      }
      delete t[i];
    }
  }

  // This function call does not return, unless, at some point in time,
  // "eventLoopWatchVariable" gets set to something non-zero.

  // If you choose to continue the application past this point (i.e., if you
  // comment out the "return 0;" statement above), and if you don't intend to do
  // anything more with the "TaskScheduler" and "UsageEnvironment" objects, then
  // you can also reclaim the (small) memory used by these objects by
  // uncommenting the following code:
  /*
    env_->reclaim(); env_ = NULL;
    delete scheduler_; scheduler_ = NULL;
  */
}

void RtspPlugin::CheckRtspState() {
  const std::vector<std::shared_ptr<MediaPipeLine>> &pipelines =
      MediaPipeManager::GetInstance().GetPipeLine();

  sleep(10);  // wait for rtsp stream connect success
  LOGW << "Start CheckRtspState thread,running flag:" << running_;
  while (running_) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t time_now = (uint64_t)tv.tv_sec;
    for (uint32_t i = 0; i < pipelines.size(); ++i) {
      if (time_now - pipelines[i]->GetlastReadDataTime() < 10) {
        continue;
      }

      int channel = pipelines[i]->GetGrpId();
      LOGE << "RTSP channel:" << channel << " , 10 seconds no stream!";
      if (rtsp_clients_[i] &&
          pipelines[i]->GetDecodeType() !=
              RTSP_Payload_NONE) {  // if not start, maybe the destructor
                                    // has been called
        rtsp_clients_[i]->Stop();
      }

      // reopen rtsp url
      ourRTSPClient *client = nullptr;
      client = openURL(*env_, "RTSPClient", rtsp_url_[i].url.c_str(),
                       rtsp_url_[i].tcp_flag, rtsp_url_[i].frame_max_size,
                       ("channel" + std::to_string(i) + ".stream"),
                       rtsp_url_[i].save_stream, i);
      LOGI << "after reopen rtsp stream, channel:" << i;
      rtsp_clients_[i] = client;
      pipelines[i]->UpdateTime();
    }
    sleep(1);
  }
}

int RtspPlugin::Start() {
  if (draw_real_time_video_) {
    int size = channel_number_ * buffer_count_ + 1;
    use_flag_list.resize(channel_number_ * buffer_count_);
    int nRet = 0;
    for (int i = 0; i < size; ++i) {
      auto pym_img = new pym_buffer_t();
      nRet = AllocPymBuffer(pym_img, running_venc_720p_ && encode_smart_);
      if (nRet != 0) {
        LOGE << "rtsplugin alloc pym fail, start fail";
        return nRet;
      }
      pym_buffers_.push_back(pym_img);
      use_flag_list[i] = false;
    }
  }
  process_thread_ = std::make_shared<std::thread>(&RtspPlugin::Process, this);
  check_thread_ =
      std::make_shared<std::thread>(&RtspPlugin::CheckRtspState, this);
  return 0;
}

int RtspPlugin::Stop() {
  LOGW << "RtspPlugin Stop";
  running_ = false;
  LOGW << "process_thread_ Stop";
  const std::vector<std::shared_ptr<MediaPipeLine>> &pipelines =
      MediaPipeManager::GetInstance().GetPipeLine();
  LOGW << "pipe line size: " << pipelines.size();
  for (uint32_t i = 0; i < pipelines.size(); ++i) {
    pipelines[i]->Stop();
  }

  eventLoopWatchVariable = 1;
  check_thread_->join();
  process_thread_->join();

  if (draw_real_time_video_) {
    int size = channel_number_ * buffer_count_ + 1;
    for (int i = 0; i < size; ++i) {
      FreePymBuffer(pym_buffers_[i], running_venc_720p_ && encode_smart_);
    }
  }
  return 0;
}

void RtspPlugin::GetConfigFromFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    LOGE << "Open config file " << path << " failed";
  }
  ifs >> config_;
  ifs.close();

  auto value_js = config_["rtsp_config_file"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: channel_num";
  }
  LOGW << value_js;
  std::string rtsp_config_file = value_js.asString();

  value_js = config_["drop_frame_config_file"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: channel_num";
  }
  LOGW << value_js;
  std::string drop_frame_config_file = value_js.asString();

  value_js = config_["display_config_file"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: channel_num";
  }
  LOGW << value_js;
  std::string display_config_file = value_js.asString();

  value_js = config_["run_smart"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: run_smart";
  } else {
    not_run_smart_ = !value_js.asBool();
  }
  LOGW << "video box config not run smart flag:" << not_run_smart_;

  GetRtspConfigFromFile(rtsp_config_file);
  GetDropFrameConfigFromFile(display_config_file);
  GetDisplayConfigFromFile(display_config_file);
}

void RtspPlugin::GetRtspConfigFromFile(const std::string &path) {
  std::ifstream rtsp_file(path);
  if (!rtsp_file.is_open()) {
    LOGE << "Open config file " << path << " failed";
    return;
  }
  rtsp_file >> config_;
  rtsp_file.close();

  auto value_js = config_["channel_num"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: channel_num";
  }
  LOGW << value_js;
  channel_number_ = value_js.asInt();
  for (int i = 0; i < channel_number_; ++i) {
    std::string channel("channel" + std::to_string(i));
    std::string rtsp_url = config_[channel.c_str()]["rtsp_link"].asString();
    // bool use_tcp = config_[channel.c_str()]["tcp"].asBool();
    LOGW << channel << ": rtsp url: " << rtsp_url;
    Rtspinfo info;
    info.url = rtsp_url;
    info.tcp_flag = config_[channel.c_str()]["tcp"].asBool();
    info.frame_max_size = config_[channel.c_str()]["frame_max_size"].asInt();
    info.save_stream = config_[channel.c_str()]["save_stream"].asBool();
    LOGW << "channel: " << channel << " protocol tcp flag: " << info.tcp_flag
         << "max frame size:" << info.frame_max_size;
    rtsp_url_.push_back(info);
  }

  rtsp_clients_stat_.resize(channel_number_);
}

void RtspPlugin::GetDropFrameConfigFromFile(const std::string &path) {
  std::ifstream drop_frame_file(path);
  if (!drop_frame_file.is_open()) {
    LOGE << "Open config file " << path << " failed";
    return;
  }
  drop_frame_file >> config_;
  drop_frame_file.close();

  auto value_js = config_["frame_drop"];
  if (value_js.isNull()) {
    LOGE << "Can not find key: frame_drop";
  }

  drop_frame_ = config_["frame_drop"]["drop_frame"].asBool();
  drop_frame_interval_ = config_["frame_drop"]["interval_frames_num"].asInt();
}

void RtspPlugin::GetDisplayConfigFromFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    LOGE << "Open config file " << path << " failed";
    return;
  }
  ifs >> config_;
  ifs.close();

  running_vot_ = config_["vo"]["enable"].asBool();
  display_mode_ = config_["vo"]["display_mode"].asInt();
  transition_support_ = config_["vo"]["transition_support"].asBool();
  draw_smart_ = config_["vo"]["draw_smart"].asBool();
  draw_real_time_video_ = config_["vo"]["draw_real_time_video"].asBool();

  running_venc_1080p_ = config_["rtsp"]["stream_1080p"].asBool();
  running_venc_720p_ = config_["rtsp"]["stream_720p"].asBool();
  encode_smart_ = config_["rtsp"]["encode_smart_info"].asBool();
}

}  // namespace rtspplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
