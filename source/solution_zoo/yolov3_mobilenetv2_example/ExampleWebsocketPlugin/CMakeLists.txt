cmake_minimum_required(VERSION 2.8)

# 项目名称
project(examplewebsocketplugin)
set(CMAKE_CXX_STANDARD 11)
option(RELEASE_LIB "build version of release" ON)

file(GLOB PROJECT_INC
    "include/*.h"
)

file(GLOB PROJECT_SRC
    "src/*.cpp"
)

set(SOURCE_FILES ${PROJECT_SRC} ${PROJECT_INC})

include_directories(
    include
    src
)

set(BASE_LIBRARIES pthread jsoncpp protobuf
    xproto_msgtype websocketplugin)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${MY_OUTPUT_ROOT}/lib)