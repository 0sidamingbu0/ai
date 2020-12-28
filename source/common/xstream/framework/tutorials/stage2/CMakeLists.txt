cmake_minimum_required(VERSION 2.8)

# 设置头文件搜索路径
include_directories(./)

set(BASE_LIBRARIES xstream hobotlog jsoncpp pthread rt)

set(SOURCE_FILES
  method/method.h
  method/value.h
  method/average_method.cc
  method/standard_deviation_method.cc
  method/method_factory.cc)

add_executable(stage2_multithread_example
  stage2_multithread.cc
  ${SOURCE_FILES})
target_link_libraries(stage2_multithread_example ${BASE_LIBRARIES})
