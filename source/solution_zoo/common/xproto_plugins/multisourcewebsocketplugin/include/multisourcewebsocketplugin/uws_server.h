/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#ifndef INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_UWS_SERVER_H_
#define INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_UWS_SERVER_H_
#include <memory>
#include <string>
#include <thread>

#include "uWS/uWS.h"
#include "xproto/threads/threadpool.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {
using hobot::CThreadPool;

class UwsServer {
 public:
  explicit UwsServer(const std::string &config);
  UwsServer() : connetion_(nullptr), worker_(nullptr) {}
  ~UwsServer() = default;

 public:
  int Init(int port);
  int Send(const std::string &protocol);
  int DeInit();

 private:
  void StartServer();
  std::mutex mutex_;
  uWS::WebSocket<uWS::SERVER> *connetion_;
  std::shared_ptr<std::thread> worker_;
 private:
  std::string config_file_;
  int port_;
};
}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // INCLUDE_MULTISOURCEWEBSOCKETPLUGIN_UWS_SERVER_H_
