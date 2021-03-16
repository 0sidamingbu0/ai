/*
 * @Description: implement of multiwebsocketplugin by J3 media lib
 * @Author: hangjun.yang@horizon.ai
 * @Date: 2020-09-01 18:00:00
 * @LastEditors: hangjun.yang@horizon.ai
 * @LastEditTime: 2020-08-22 19:00:00
 * @Copyright 2017~2020 Horizon Robotics, Inc.
 */

#include "multisourcewebsocketplugin/uws_server.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include "hobotlog/hobotlog.hpp"
#include "uWS/uWS.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace multisourcewebsocketplugin {
using std::chrono::milliseconds;

int UwsServer::Init(int port) {
  if (nullptr == worker_) {
    worker_ = std::make_shared<std::thread>(&UwsServer::StartServer, this);
    worker_->detach();
  }
  port_ = port;
  return 0;
}

int UwsServer::DeInit() {
  //  server_.
  LOGI <<"UwsServer::DeInit()";
  connetion_ = nullptr;
  return 0;
}

UwsServer::UwsServer(const std::string &config_file) : connetion_(nullptr),
  worker_(nullptr) {
}

void UwsServer::StartServer() {
  LOGI << "UwsServer::StartServer";
  uWS::Hub hub;
  hub.onConnection(
      [this](uWS::WebSocket<uWS::SERVER> *ws, uWS::HttpRequest req) {
        LOGI << "UwsServer Connection with PC success";
        std::lock_guard<std::mutex> connection_mutex(mutex_);
        connetion_ = ws;
      });

  hub.onMessage([this](uWS::WebSocket<uWS::SERVER> *ws, char *message,
                       size_t length, uWS::OpCode opCode) {
    LOGI << "UwsServer onMessage: " << message;
  });

  hub.onDisconnection([this](uWS::WebSocket<uWS::SERVER> *ws, int code,
                             char *message, size_t length) {
    std::lock_guard<std::mutex> connection_mutex(mutex_);
    connetion_ = nullptr;
    LOGI << "UwsServer Disconnection with PC success";
  });
  if (!hub.listen(port_)) {
    LOGI << "UwsServer start failed";
    return;
  }
  LOGI << "UwsServer begin to run";
  hub.run();
}
int UwsServer::Send(const std::string &protocol) {
  if (connetion_ != nullptr) {
    LOGI << "UwsServer begin send protocol";
    connetion_->send(protocol.c_str(), protocol.size(), uWS::OpCode::BINARY);
    LOGI << "UwsServer send protocol size = " << protocol.size();
  }
  return 0;
}
}  // namespace multisourcewebsocketplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
