cmake_minimum_required(VERSION 2.8)

# project name
project(xproto)

option(PARENT_BUILD "is build from parent" OFF)
if (NOT ${PARENT_BUILD})
  get_filename_component(XPROTO_CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  #message("xproto current dir init:${XPROTO_CURRENT_DIR}")
  set(RELEASE_ROOT_DIR)
  function(get_parent_dir arg dir n)
    foreach(i RANGE ${ARGV2})
      get_filename_component(dir ${dir} DIRECTORY)
      #message("i:${i} dir:${dir}")
    endforeach()
    set(RELEASE_ROOT_DIR ${dir} PARENT_SCOPE)
  endfunction()

  get_parent_dir(XPROTO_CURRENT_DIR ${XPROTO_CURRENT_DIR} 2)
  option(X86_ARCH "x86 architecture" OFF)
  option(X86_CENTOS "centos release" OFF)
endif()

option(RELEASE_LIB "build version of release" OFF)

if (NOT ${PARENT_BUILD})
  if (${RELEASE_LIB})
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "BUILD TYPE:Release")
  else ()
    set(CMAKE_BUILD_TYPE Debug)
    set(HOBOT_BUILD_DEBUG "TRUE")
    message(STATUS "BUILD TYPE:Debug")
  endif ()

  list(APPEND CMAKE_C_FLAGS "-Wall -Wno-unknown-pragmas -fPIC -fexceptions -Werror ")
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    list(APPEND CMAKE_C_FLAGS "-O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC ")
  else ()
    list(APPEND CMAKE_C_FLAGS "-g -O0 -fPIC ")
  endif ()

  # x86 architecture
  if (${X86_ARCH})
    # centos release
    if (${X86_CENTOS})
      message("tool chain:${CMAKE_CXX_COMPILER}")
      add_definitions(-std=c++11)
      include_directories(
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/centos/hobotlog/include
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/centos/gtest/include
      )
      link_directories(
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/centos/hobotlog/lib
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/centos/gtest/lib
      )
    else ()
      message("tool chain:${CMAKE_CXX_COMPILER}")
      add_definitions(-std=c++11)
      include_directories(
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/ubuntu/hobotlog/include
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/ubuntu/gtest/include
      )
      link_directories(
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/ubuntu/hobotlog/lib
        ${XPROTO_CURRENT_DIR}/framework/third_party/x86/ubuntu/gtest/lib
      )
    endif ()
  # arm architecture
  else ()
    set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
    message("tool chain:${CMAKE_CXX_COMPILER}")
    include_directories(
      ${XPROTO_CURRENT_DIR}/framework/third_party/aarch64/hobotlog/include
      ${XPROTO_CURRENT_DIR}/framework/third_party/aarch64/gtest/include
    )
    link_directories(
      ${XPROTO_CURRENT_DIR}/framework/third_party/aarch64/hobotlog/lib
      ${XPROTO_CURRENT_DIR}/framework/third_party/aarch64/gtest/lib
    )
  endif ()

  # build separately install directory
  set(OUTPUT_ROOT ${CMAKE_SOURCE_DIR}/output/)
endif ()

# add header files path
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

# header files
set(PROJECT_INC
  "include/xproto/manager/msg_manager.h"
  "include/xproto/message/pluginflow/flowmsg.h"
  "include/xproto/message/pluginflow/msg_registry.h"
  "include/xproto/plugin/xplugin.h"
  "include/xproto/plugin/xpluginasync.h"
  "include/xproto/threads/threadpool.h"
)

# source files
set(PROJECT_SRC
  "src/message/pluginflow/msg_registry.cpp"
  "src/plugin/xplugin.cpp"
  "src/plugin/xpluginasync.cpp"
  "src/threads/threadpool.cpp"
  "src/utils/gtest_helper.cpp"
  "src/utils/time_helper.cpp"
)

set(SOURCE_FILES
  ${PROJECT_SRC}
  ${PROJECT_INC}
)

option(BUILD_SHARED_LIBS "build shared libs" OFF)
add_library(${PROJECT_NAME} ${SOURCE_FILES})

add_subdirectory(test)
add_subdirectory(tutorials/stage1)
add_subdirectory(tutorials/stage3)
add_subdirectory(tutorials/stage2)

# make install
set(OUTPUT_ROOT ${PROJECT_SOURCE_DIR}/output)
set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/xproto)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/manager/msg_manager.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/xproto/manager
)

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/plugin/xplugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/plugin/xpluginasync.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/xproto/plugin
)

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/message/pluginflow/flowmsg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/message/pluginflow/msg_registry.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/xproto/message/pluginflow
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/threads/threadpool.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/xproto/threads
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/xproto/utils/singleton.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/xproto/utils
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  DESTINATION ${MY_OUTPUT_ROOT}/
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/document
  DESTINATION ${MY_OUTPUT_ROOT}/
)

install(TARGETS ${PROJECT_NAME}
	DESTINATION ${MY_OUTPUT_ROOT}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tutorials
	DESTINATION ${MY_OUTPUT_ROOT}/
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party
	DESTINATION ${MY_OUTPUT_ROOT}/tutorials
)
