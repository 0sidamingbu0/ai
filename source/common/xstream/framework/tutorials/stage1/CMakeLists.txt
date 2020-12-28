cmake_minimum_required(VERSION 2.8)

# 设置头文件搜索路径
include_directories(./)

set(BASE_LIBRARIES xstream hobotlog jsoncpp pthread rt)

set(SOURCE_FILES
  method/bbox.h
  method/bbox_filter.h
  method/method_factory.cc)

add_executable(stage1_bbox_filter_example
  bbox_filter_main.cc
  ${SOURCE_FILES})
target_link_libraries(stage1_bbox_filter_example ${BASE_LIBRARIES})


