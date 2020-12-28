/**
* @copyright Copyright (c) 2020 Horizon Robotics. All rights reserved.
* @file      main.cc
* @brief     simple example of method
* @author    zhe.sun
* @date      2020/12/24
*/

#include <iostream>
#include <string>
#include <thread>
#include "hobotxsdk/xstream_data.h"
#include "horizon/vision_type/vision_type.hpp"
#include "hobotlog/hobotlog.hpp"
#include "opencv2/opencv.hpp"
#include "hobotxsdk/xstream_sdk.h"
#ifdef X3
#include "./vio_wrapper_global.h"
#endif


// 回调函数，用于异步运行
void OnCallback(xstream::OutputDataPtr output) {
  // TODO(sz)
}

int main(int argc, char const *argv[]) {
  if (argc < 4) {
    std::cout << "Usage : ./stage10_example jpg_img_path "
              << "vio_config workflow_config" << std::endl;
    std::cout << "Example : ./stage10_example ./config/1080p.jpg "
              << "./config/vio_config.json ./config/workflow.json"
              << std::endl;
    return -1;
  }
#ifdef X2
  HOBOT_CHECK(0) << "not support x2 platform";
#endif
  SetLogLevel(HOBOT_LOG_DEBUG);

  std::string input_image = argv[1];
  std::string vio_config = argv[2];
  std::string workflow_config = argv[3];

  // 0. 创建sdk
  xstream::XStreamSDK *flow = xstream::XStreamSDK::CreateSDK();
  // Set config_file
  flow->SetConfig("config_file", workflow_config);
  flow->Init();
  // Set callback
  flow->SetCallback(&OnCallback);

#ifdef X3
  HbVioFbWrapperGlobal fb_vio(vio_config);
  auto ret = fb_vio.Init();
  HOBOT_CHECK(ret == 0) << "fb vio init failed!!!";

  cv::Mat bgr_img = cv::imread(input_image);
  int width = bgr_img.cols;
  int height = bgr_img.rows;
  cv::Mat img_nv12;
  // bgr to nv12
  {
    cv::Mat img_yuv;
    cv::cvtColor(bgr_img, img_yuv, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = img_yuv.ptr<uint8_t>();
    img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
    uint8_t *nv12 = img_nv12.ptr<uint8_t>();
    int uv_height = height / 2;
    int uv_width = width / 2;
    // copy y data
    int y_size = uv_height * uv_width * 4;
    memcpy(nv12, yuv, y_size);

    // copy uv data
    int uv_stride = uv_width * uv_height;
    uint8_t *uv_data = nv12 + y_size;
    for (int i = 0; i < uv_stride; ++i) {
      *(uv_data++) = *(yuv + y_size + i);
      *(uv_data++) = *(yuv + y_size + +uv_stride + i);
    }
  }

  auto py_image_frame_ptr = fb_vio.GetImgInfo(img_nv12.data, width, height);
  HOBOT_CHECK(py_image_frame_ptr != nullptr) << "fb vio get image failed!!!";

  // 1. 准备输入金字塔数据(pyramid_img)
  xstream::InputDataPtr inputdata(new xstream::InputData());
  auto xstream_pyramid = std::make_shared<xstream::XStreamData<
      std::shared_ptr<hobot::vision::ImageFrame>>>();
  xstream_pyramid->value = py_image_frame_ptr;
  xstream_pyramid->name_ = "image";   // 需要与workflow中对应input数据name一致
  inputdata->datas_.push_back(xstream_pyramid);

  // 2. 异步运行
  flow->AsyncPredict(inputdata);
#endif

  std::this_thread::sleep_for(std::chrono::seconds(1));
  delete flow;
  return 0;
}
