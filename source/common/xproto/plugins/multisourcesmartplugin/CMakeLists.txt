cmake_minimum_required(VERSION 2.8)

# 项目名称
project(multisourcesmartplugin)

# 添加头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# 头文件路径信息
file(GLOB PROJECT_INC
    "include/multisourcesmartplugin/*.h"
)


file(GLOB PROJECT_SRC
    "src/multisourcesmartplugin/*.cpp")


set(SOURCE_FILES ${PROJECT_SRC} ${PROJECT_INC})
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set(BASE_LIBRARIES hobotlog opencv_world pthread  commongdcplugin xproto_msgtype jsoncpp xstream-imagetools xstream xproto)

target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${MY_OUTPUT_ROOT}/include/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${MY_OUTPUT_ROOT}/lib)
# install configs, some scripts will depends this
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/config/
    DESTINATION ${MY_OUTPUT_ROOT}/config/)

