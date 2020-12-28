cmake_minimum_required(VERSION 2.8)
project(xstream
	LANGUAGES CXX
)

option(X86_ARCH "x86 architecture" OFF)
option(X86_CENTOS "centos release" OFF)

get_filename_component(XSTREAM_TUTORIALS_CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
# message("xstream release dir init:${XSTREAM_TUTORIALS_CURRENT_DIR}")

# x86 architecture
if (${X86_ARCH})
  if (${X86_CENTOS})
    message("tool chain:${CMAKE_CXX_COMPILER}")
    add_definitions(-std=c++11)
    include_directories(
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/jsoncpp/include/
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/hobotlog/include/
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/vision_type/include/
    )
    link_directories(
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/jsoncpp/lib
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/hobotlog/lib
    )
  else ()
    message("tool chain:${CMAKE_CXX_COMPILER}")
    add_definitions(-std=c++11)
    include_directories(
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/jsoncpp/include/
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/hobotlog/include/
    )
    link_directories(
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/hobotlog/lib
    )
  endif ()
else ()
  set(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
  message("tool chain:${CMAKE_CXX_COMPILER}")
  include_directories(
    ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/jsoncpp/include/
    ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/hobotlog/include/
    ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/opencv/include/
    ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/gtest/include/
  )
  link_directories(
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/hobotlog/lib
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/jsoncpp/lib
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/gtest/lib
    )
endif ()

include_directories(
  ${XSTREAM_TUTORIALS_CURRENT_DIR}/include
)

link_directories(
  ${XSTREAM_TUTORIALS_CURRENT_DIR}/lib
)

add_subdirectory(stage1)
add_subdirectory(stage2)
add_subdirectory(stage3)
add_subdirectory(stage4)
add_subdirectory(stage5)
add_subdirectory(stage6)
add_subdirectory(stage7)
add_subdirectory(stage8)
add_subdirectory(stage9)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage1/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage1
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage2/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage2
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage3/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage3
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage4/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage4
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage5/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage5
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage6/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage6
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage7/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage7
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage8/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage8
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stage9/config
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/stage9
)
