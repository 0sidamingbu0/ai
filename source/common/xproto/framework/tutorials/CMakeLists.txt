cmake_minimum_required(VERSION 2.8)
project(xproto
	LANGUAGES CXX
)

option(X86_ARCH "x86 architecture" OFF)
option(X86_CENTOS "centos release" OFF)

get_filename_component(XPROTO_TUTORIALS_CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
# message("xproto release dir init:${XPROTO_TUTORIALS_CURRENT_DIR}")

# x86 architecture
if (${X86_ARCH})
  if (${X86_CENTOS})
    message("tool chain:${CMAKE_CXX_COMPILER}")
    add_definitions(-std=c++11)
    include_directories(
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/jsoncpp/include/
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/hobotlog/include/
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/vision_type/include/
    )
    link_directories(
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/jsoncpp/lib
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/hobotlog/lib
    )
  else ()
    message("tool chain:${CMAKE_CXX_COMPILER}")
    add_definitions(-std=c++11)
    include_directories(
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/gcc_${CMAKE_CXX_COMPILER_VERSION}/hobotlog/include/
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/gcc_${CMAKE_CXX_COMPILER_VERSION}/jsoncpp/include/
    )
    link_directories(
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/gcc_${CMAKE_CXX_COMPILER_VERSION}/hobotlog/lib/
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/gcc_${CMAKE_CXX_COMPILER_VERSION}/jsoncpp/lib/
    )
  endif ()
else ()
  set(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
  message("tool chain:${CMAKE_CXX_COMPILER}")
  include_directories(
    ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/jsoncpp/include/
    ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/hobotlog/include/
    ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/opencv/include/
    ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/gtest/include/
  )
  link_directories(
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/hobotlog/lib
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/jsoncpp/lib
      ${XPROTO_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/gtest/lib
    )
endif ()

include_directories(
  ${XPROTO_TUTORIALS_CURRENT_DIR}/include
)

link_directories(
  ${XPROTO_TUTORIALS_CURRENT_DIR}/lib
)

add_subdirectory(stage1)
add_subdirectory(stage2)
add_subdirectory(stage3)
