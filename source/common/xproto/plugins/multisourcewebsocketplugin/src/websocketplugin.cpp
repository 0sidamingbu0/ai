/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */
#include "multisourcewebsocketplugin/websocketplugin.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <map>
#include <vector>

#include "hobotlog/hobotlog.hpp"
#include "multisourcewebsocketplugin/convert.h"
#include "xproto/message/pluginflow/msg_registry.h"
#include "xproto_msgtype/protobuf/x3.pb.h"
#include "xproto_msgtype/smartplugin_data.h"
#include "xproto_msgtype/vioplugin_data.h"
#include "utils/time_helper.h"

using hobot::Timer;
namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {
using horizon::vision::xproto::XPluginErrorCode;
using horizon::vision::xproto::basic_msgtype::SmartMessage;
using horizon::vision::xproto::basic_msgtype::VioMessage;
using std::chrono::milliseconds;

WebsocketPlugin::WebsocketPlugin(std::string config_file) {
  config_file_ = config_file;
  LOGI << "UwsPlugin smart config file:" << config_file_;
  smart_stop_flag_ = false;
  video_stop_flag_ = false;
  Reset();
}

WebsocketPlugin::~WebsocketPlugin() {
  config_ = nullptr;
}

int WebsocketPlugin::Init() {
  LOGI << "WebsocketPlugin::Init";
  // load config
  config_ = std::make_shared<WebsocketConfig>(config_file_);
  if (!config_ || !config_->LoadConfig()) {
    LOGE << "failed to load config file";
    return -1;
  }

  if (config_->display_mode_ != WebsocketConfig::WEB_MODE) {
    LOGF << "only support web display mode, please set mode=1";
    return 0;
  }

  RegisterMsg(GetSmartMessageType(), std::bind(&WebsocketPlugin::FeedSmart,
                                               this, std::placeholders::_1));
  RegisterMsg(TYPE_IMAGE_MESSAGE, std::bind(&WebsocketPlugin::FeedVideo, this,
                                            std::placeholders::_1));
  jpg_encode_thread_.CreatThread(1);
  data_send_thread_.CreatThread(1);
  // 调用父类初始化成员函数注册信息
  XPluginAsync::Init();
  return 0;
}

int WebsocketPlugin::Reset() {
  LOGI << __FUNCTION__ << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
  std::unique_lock<std::mutex> lock(map_smart_mutex_);

  for (auto itr = x3_frames_.begin(); itr != x3_frames_.end(); ++itr) {
    auto &frames = itr->second;
    while (!frames.empty()) {
      frames.pop();
    }
  }

  for (auto itr = x3_smart_msg_.begin(); itr != x3_smart_msg_.end(); ++itr) {
    auto &smarts = itr->second;
    while (!smarts.empty()) {
      smarts.pop();
    }
  }
  x3_frames_.clear();
  x3_smart_msg_.clear();
  return 0;
}

int WebsocketPlugin::Start() {
  if (config_->display_mode_ != WebsocketConfig::WEB_MODE) {
    LOGF << "not support web display";
    return 0;
  }
  for (size_t index = 0; index < config_->chnn_list_.size(); ++index) {
    int ch_id = config_->chnn_list_[index];
    int port = 8080 + 2 * index;
    uws_server_[ch_id] = std::make_shared<UwsServer>();
    if (uws_server_[ch_id]->Init(port) != 0) {
      LOGF << "websocket plugin Init uWS server failed";
      return -1;
    }
    video_encode_chn_[ch_id] = 1;  //  init default chn
  }

  /* 1. media codec init */
  /* 1.1 get media codec manager and module init */
  MediaCodecManager &manager = MediaCodecManager::Get();
  auto rv = manager.ModuleInit();
  HOBOT_CHECK(rv == 0);
  for (auto iter = video_encode_chn_.begin();
       iter != video_encode_chn_.end(); iter++) {
    /* 1.2 get media codec venc chn */
    auto encode_chn = manager.GetEncodeChn();
    /* 1.3 media codec venc chn init */
    int pic_width = config_->image_width_;
    int pic_height = config_->image_height_;
    int frame_buf_depth = config_->frame_buf_depth_;
    int is_cbr = config_->is_cbr_;
    int bitrate = config_->bitrate_;

    rv = manager.EncodeChnInit(encode_chn, PT_JPEG, pic_width, pic_height,
                               frame_buf_depth, HB_PIXEL_FORMAT_NV12, is_cbr,
                               bitrate);
    HOBOT_CHECK(rv == 0);
    /* 1.4 set media codec venc jpg chn qfactor params */
    rv = manager.SetUserQfactorParams(encode_chn, config_->jpeg_quality_);
    HOBOT_CHECK(rv == 0);
    /* 1.5 set media codec venc jpg chn qfactor params */
    rv = manager.EncodeChnStart(encode_chn);
    HOBOT_CHECK(rv == 0);
    /* 1.6 alloc media codec vb buffer init */
    if (config_->use_vb_) {
      int vb_num = frame_buf_depth;
      int pic_stride = config_->image_width_;
      int pic_size = pic_stride * pic_height * 3 / 2;  // nv12 format
      int vb_cache_enable = 1;
      rv = manager.VbBufInit(encode_chn, pic_width, pic_height, pic_stride,
                             pic_size, vb_num, vb_cache_enable);
      HOBOT_CHECK(rv == 0);
    }
    iter->second = encode_chn;
    LOGD << "channel id = " << iter->first
         << ", encode chnid = " << iter->second;
  }

  if (!worker_) {
    worker_ = std::make_shared<std::thread>(
        std::bind(&WebsocketPlugin::MapSmartProc, this));
  }
  return 0;
}

void WebsocketPlugin::MapSmartProc() {
  static std::map<int, uint64_t> pre_frame_id;
  while (!map_stop_) {
    std::unique_lock<std::mutex> lock(map_smart_mutex_);
    map_smart_condition_.wait(lock);
    if (map_stop_) {
      break;
    }
    if (!x3_smart_msg_.empty() && !x3_frames_.empty()) {
      for (auto smart_itr = x3_smart_msg_.begin();
           smart_itr != x3_smart_msg_.end(); ++smart_itr) {
        int ch_id = smart_itr->first;
        if (smart_itr->second.empty()) {
          continue;
        }
        if (x3_frames_.count(ch_id) == 0 || x3_frames_[ch_id].empty()) {
          continue;
        }
        auto &smarts = smart_itr->second;
        auto &iamges = x3_frames_[ch_id];

        auto msg = smarts.top();
        auto frame = iamges.top();
        if (msg->time_stamp == frame.timestamp_()) {
          if (pre_frame_id.count(ch_id) == 0) {
            pre_frame_id[ch_id] = 0;
          }
          if (msg->frame_id > pre_frame_id[ch_id] ||
              (pre_frame_id[ch_id] - msg->frame_id > 300) ||
              pre_frame_id[ch_id] == 0) {  // frame_id maybe overflow reset to 0
            int task_num = data_send_thread_.GetTaskNum();
            if (task_num < 12) {
              data_send_thread_.PostTask(std::bind(
                  &WebsocketPlugin::SendSmartMessage, this, msg, frame));
            }
            pre_frame_id[ch_id] = msg->frame_id;
          }
          smarts.pop();
          iamges.pop();
          // break;
        } else {
          // avoid smart or image result lost
          while (smarts.size() > 5) {
            auto msg_inner = smarts.top();
            auto frame_inner = iamges.top();
            if (msg_inner->time_stamp < frame_inner.timestamp_()) {
              // 消息对应的图片一直没有过来，删除消息
              smarts.pop();
            } else {
              break;
            }
          }
          while (iamges.size() > 5) {
            auto msg_inner = smarts.top();
            auto frame_inner = iamges.top();
            if (frame_inner.timestamp_() < msg_inner->time_stamp) {
              // 图像对应的消息一直没有过来，删除图像
              iamges.pop();
            } else {
              break;
            }
          }
          break;
        }
      }
    }
  }
}

int WebsocketPlugin::Stop() {
  {
    std::lock_guard<std::mutex> smart_lock(smart_mutex_);
    smart_stop_flag_ = true;
  }
  {
    std::lock_guard<std::mutex> video_lock(video_mutex_);
    video_stop_flag_ = true;
  }
  {
    if (worker_ && worker_->joinable()) {
      map_stop_ = true;
      map_smart_condition_.notify_one();
      worker_->join();
      worker_ = nullptr;
      LOGI << "WebsocketPlugin stop worker";
    }
  }
  LOGI << "WebsocketPlugin::Stop()";
  /* 3. media codec deinit */
  MediaCodecManager &manager = MediaCodecManager::Get();
  for (auto iter = video_encode_chn_.begin(); iter != video_encode_chn_.end();
       iter++) {
    auto chn = iter->first;
    /* 3.1 media codec chn stop */
    manager.EncodeChnStop(chn);
    /* 3.2 media codec chn deinit */
    manager.EncodeChnDeInit(chn);
    /* 3.3 media codec vb buf deinit */
    if (config_->use_vb_) {
      manager.VbBufDeInit(chn);
    }
  }
  /* 3.4 media codec module deinit */
  manager.ModuleDeInit();
  for (auto itr = uws_server_.begin(); itr != uws_server_.end(); ++itr) {
    itr->second->DeInit();
  }
  return 0;
}

int WebsocketPlugin::FeedSmart(XProtoMessagePtr msg) {
  {
    std::lock_guard<std::mutex> smart_lock(smart_mutex_);
    if (smart_stop_flag_) {
      LOGD << "Aleardy stop, WebsocketPLugin FeedSmart return";
      return -1;
    }
  }
  auto smart_msg = std::static_pointer_cast<SmartMessage>(msg);
  if (smart_msg) {
    int ch_id = smart_msg->channel_id;
    LOGI << "[websocket plugin] got one smart result, ch_id:" << ch_id;
    {
      std::lock_guard<std::mutex> smart_lock(map_smart_mutex_);
      x3_smart_msg_[ch_id].push(smart_msg);
    }
    map_smart_condition_.notify_one();
  }
  return 0;
}

int WebsocketPlugin::FeedVideo(XProtoMessagePtr msg) {
  auto frame = std::dynamic_pointer_cast<VioMessage>(msg);
  int image_num = frame->image_.size();
  for (int image_idx = 0; image_idx < image_num; image_idx++) {
    jpg_encode_thread_.PostTask(
        std::bind(&WebsocketPlugin::EncodeJpg, this, msg, image_idx));
  }
  return 0;
}

int WebsocketPlugin::SaveYuvToFile(int y_len, int uv_len, uint8_t *y_vaddr,
                                   uint8_t *uv_vaddr, std::string name) {
  FILE *fd = fopen(name.c_str(), "wb+");
  if (!fd) {
    LOGE << "open file name:%s failure!";
    return -1;
  }
  LOGD << "filename = " << name << ", y_len = " << y_len
       << ", uv_len = " << uv_len
       << ", y_vaddr = " << reinterpret_cast<int64_t>(y_vaddr)
       << ", uv_vaddr = " << reinterpret_cast<int64_t>(uv_vaddr);

  uint8_t *img_ptr = static_cast<uint8_t *>(malloc(y_len + uv_len));
  memcpy(img_ptr, y_vaddr, y_len);
  memcpy(img_ptr + y_len, uv_vaddr, uv_len);

  fwrite(img_ptr, sizeof(char), y_len + uv_len, fd);
  fflush(fd);
  fclose(fd);
  free(img_ptr);

  return 0;
}

void WebsocketPlugin::EncodeJpg(XProtoMessagePtr msg, uint32_t image_idx) {
  int rv;
  cv::Mat yuv_img;
  std::vector<uchar> img_buf;
  x3::FrameMessage x3_frame_msg;
  std::string smart_result;
  iot_venc_src_buf_t *frame_buf = nullptr;
  iot_venc_src_buf_t src_buf = { 0 };
  {
    std::lock_guard<std::mutex> video_lock(video_mutex_);
    if (video_stop_flag_) {
      LOGD << "Aleardy stop, WebsocketPLugin Feedvideo return";
      return;
    }
  }
  LOGI << "WebsocketPLugin Feedvideo";
  auto frame = std::dynamic_pointer_cast<VioMessage>(msg);
  VioMessage *vio_msg = frame.get();
  auto timestamp = vio_msg->image_[image_idx]->time_stamp;
  int channel_id = vio_msg->image_[image_idx]->channel_id;
  int encode_chn = 1;  //  default 1
  if (video_encode_chn_.find(channel_id) != video_encode_chn_.end()) {
    encode_chn = video_encode_chn_[channel_id];
  }
  LOGI << "[websocket plugin] got one video, ch_id:" << channel_id
       << ", encode_chn = " << encode_chn << ", time_stamp = " << timestamp;
  // get pyramid size
  auto pym_image = vio_msg->image_[image_idx];
  origin_image_width_ = pym_image->down_scale[0].width;
  origin_image_height_ = pym_image->down_scale[0].height;
  dst_image_width_ = pym_image->down_scale[config_->layer_].width;
  dst_image_height_ = pym_image->down_scale[config_->layer_].height;

  /* 2. start encode yuv to jpeg */
  /* 2.1 get media codec vb buf for store src yuv data */
  MediaCodecManager &manager = MediaCodecManager::Get();
  frame_buf = &src_buf;
  memset(frame_buf, 0x00, sizeof(iot_venc_src_buf_t));
  frame_buf->frame_info.pts = frame->time_stamp_;
  /* 2.2 get src yuv data */
  rv = Convertor::GetYUV(frame_buf, vio_msg, config_->layer_, image_idx);
  if (rv != 0) {
    LOGF << "Convertor::GetYUV failed";
    return;
  }
  /* 2.3. encode yuv data to jpg */
  auto ts0 = Timer::current_time_stamp();
  /* dump yuv picture */
  {
    bool enable_dump = false;
    {
      std::fstream f_try;
      f_try.open("enable_yuv.tmp", std::ios::in);
      if (f_try) {
        enable_dump = true;
        f_try.close();
      }
    }
    if (enable_dump) {
      static int frame_id[8] = {0};
      std::string file_name = "out_stream_" + std::to_string(channel_id) + "_" +
                              std::to_string(frame_id[channel_id]++) + ".yuv";
      uint32_t stride, height;
      uint32_t size, size_y, size_uv;
      uint8_t *vir_ptr0, *vir_ptr1;
      size = src_buf.frame_info.size;
      height = src_buf.frame_info.height;
      stride = src_buf.frame_info.stride;
      size_y = stride * height;
      size_uv = size - size_y;
      vir_ptr0 = reinterpret_cast<uint8_t *>(src_buf.frame_info.vir_ptr[0]);
      vir_ptr1 = reinterpret_cast<uint8_t *>(src_buf.frame_info.vir_ptr[1]);
      SaveYuvToFile(size_y, size_uv, vir_ptr0, vir_ptr1, file_name);
    }
  }
  rv = manager.EncodeYuvToJpg(encode_chn, frame_buf, img_buf);
  if (config_->jpg_encode_time_ == 1) {
    auto ts1 = Timer::current_time_stamp();
    LOGW << "******Encode yuv to jpeg cost: " << ts1 - ts0 << "ms";
  }
  if (rv == 0) {
    LOGI << "encode jpg success";
    auto image = x3_frame_msg.mutable_img_();
    x3_frame_msg.set_timestamp_(timestamp);
    image->set_buf_((const char *)img_buf.data(), img_buf.size());
    image->set_type_("jpeg");
    image->set_width_(dst_image_width_);
    image->set_height_(dst_image_height_);
    {
      std::lock_guard<std::mutex> lock(map_smart_mutex_);
      x3_frames_[channel_id].push(x3_frame_msg);
      if (x3_frames_[channel_id].size() > cache_size_)
        LOGW << "the cache is full, maybe the encode thread is slowly";
    }
    map_smart_condition_.notify_one();
    /* dump jpg picture */
    {
      bool enable_dump = false;
      {
        std::fstream f_try;
        f_try.open("enable_jpg.tmp", std::ios::in);
        if (f_try) {
          enable_dump = true;
          f_try.close();
        }
      }
      if (enable_dump || config_->dump_jpg_num_-- > 0) {
        static int frame_id[8] = {0};
        std::string file_name = "out_stream_" + std::to_string(channel_id) +
                                "_" + std::to_string(frame_id[channel_id]++) +
                                ".jpg";
        std::fstream fout(file_name, std::ios::out | std::ios::binary);
        fout.write((const char *)img_buf.data(), img_buf.size());
        fout.close();
      }
    }
  } else {
    LOGE << "X3 media codec jpeg encode failed!";
  }
  return;
}

int WebsocketPlugin::SendSmartMessage(SmartMessagePtr smart_msg,
                                      x3::FrameMessage &fm) {
  std::string protocol;
  Convertor::PackSmartMsg(protocol, smart_msg.get(),
                          origin_image_width_, origin_image_height_,
                          dst_image_width_, dst_image_height_);
  // sync video & smart frame
  x3::FrameMessage msg_send;
  msg_send.ParseFromString(protocol);
  msg_send.mutable_img_()->CopyFrom(fm.img_());
  // add system info
  std::string cpu_rate_file =
      "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq";
  std::string temp_file = "/sys/class/thermal/thermal_zone0/temp";
  std::ifstream ifs(cpu_rate_file.c_str());
  if (!ifs.is_open()) {
    LOGF << "open config file " << cpu_rate_file << " failed";
    return -1;
  }
  std::stringstream ss;
  std::string str;
  ss << ifs.rdbuf();
  ss >> str;
  ifs.close();
  auto Statistics_msg_ = msg_send.mutable_statistics_msg_();
  auto attrs = Statistics_msg_->add_attributes_();
  attrs->set_type_("cpu");
  attrs->set_value_string_(str.c_str());

  ifs.clear();
  ss.clear();
  ifs.open(temp_file.c_str());
  ss << ifs.rdbuf();
  ss >> str;

  auto temp_attrs = Statistics_msg_->add_attributes_();
  temp_attrs->set_type_("temp");
  temp_attrs->set_value_string_(str.c_str());
  std::string proto_send;
  msg_send.SerializeToString(&proto_send);
  int channel_id = smart_msg->channel_id;
  auto websocket_channel_id_str =
      getenv("websocket_channel_id");  // used to switch channel
  if (websocket_channel_id_str) {
    int websocket_channel_id = std::stoi(websocket_channel_id_str);
    if (websocket_channel_id == channel_id) {
      channel_id = 0;
    } else {
      channel_id = -1;
    }
  }
  if (uws_server_.count(channel_id) == 0) {
    // LOGF << "channel id has no websocket server, channel_id:" << channel_id;
  } else {
    uws_server_[channel_id]->Send(proto_send);
  }
  return XPluginErrorCode::ERROR_CODE_OK;
}

}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
