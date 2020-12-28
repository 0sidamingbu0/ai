cmake_minimum_required(VERSION 2.8)

# 项目名称
project(examplesmartplugin)

# 添加头文件路径
include_directories(
    include
)

file(GLOB PROJECT_SRC
    "src/*.cpp"
)

add_library(${PROJECT_NAME} SHARED ${PROJECT_SRC})

set(BASE_LIBRARIES xstream xproto xproto_msgtype
    vision_type_util smartplugin hobotlog jsoncpp protobuf)
target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${MY_OUTPUT_ROOT}/lib)