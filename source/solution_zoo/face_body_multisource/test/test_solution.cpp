/*!
 * Copyright (c) 2016-present, Horizon Robotics, Inc.
 * All rights reserved.
 * \File     test_solution.cpp
 * \Author   Zhuoran Rong
 * \Mail     zhuoran.rong@horizon.ai
 * \Version  1.0.0.0
 * \Date     2020/04/18
 * \Brief    implement of test_solution.cpp
 */
#include "hobotlog/hobotlog.hpp"
#include "smartplugin/smartplugin_multisource.h"
#include "gtest/gtest.h"
#include <fstream>
#include <string>
#include <sys/utsname.h>


// declear actual main
int solution_main(int argc, const char **argv);

namespace {

// normal run
TEST(MultiSourceSolution, normal) {
  int ret = 0;
  const char *argv[] = {
      "face_body_multisource_test",                              // argv0
      "./configs/vio_config.json.96board.hg",                    // argv1
      "./face_body_multisource/configs/face_body_solution.json",  // argv2
      "-i",                                                      // argv3
      "ut",                                                      // normal
  };
  ret = solution_main(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);

  argv[3] = "-g";  // unknow parameter
  ret = solution_main(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
}

}  // namespace
