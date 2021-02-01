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
    set(CENTOS_THIRD_PARTY_DIR ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/centos/)
    include_directories(
      ${CENTOS_THIRD_PARTY_DIR}jsoncpp/include/
      ${CENTOS_THIRD_PARTY_DIR}hobotlog/include/
      ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/vision_type/include/
    )
    link_directories(
      ${CENTOS_THIRD_PARTY_DIR}jsoncpp/lib/
      ${CENTOS_THIRD_PARTY_DIR}hobotlog/lib/
    )
  else ()
    message("tool chain:${CMAKE_CXX_COMPILER}")
    add_definitions(-std=c++11)
    set(UBUNTU_THIRD_PARTY_DIR ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/x86/ubuntu/gcc_${CMAKE_CXX_COMPILER_VERSION}/)
    include_directories(
      ${UBUNTU_THIRD_PARTY_DIR}jsoncpp/include/
      ${UBUNTU_THIRD_PARTY_DIR}hobotlog/include/
    )
    link_directories(
      ${UBUNTU_THIRD_PARTY_DIR}jsoncpp/lib/
      ${UBUNTU_THIRD_PARTY_DIR}hobotlog/lib/
    )
  endif ()
else ()
  set(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
  message("tool chain:${CMAKE_CXX_COMPILER}")
  set(AARCH64_THIRD_PARTY_DIR ${XSTREAM_TUTORIALS_CURRENT_DIR}/tutorials/third_party/aarch64/)
  include_directories(
    ${AARCH64_THIRD_PARTY_DIR}jsoncpp/include/
    ${AARCH64_THIRD_PARTY_DIR}hobotlog/include/
    ${AARCH64_THIRD_PARTY_DIR}opencv/include/
    ${AARCH64_THIRD_PARTY_DIR}gtest/include/
  )
  link_directories(
      ${AARCH64_THIRD_PARTY_DIR}jsoncpp/lib/
      ${AARCH64_THIRD_PARTY_DIR}hobotlog/lib/
      ${AARCH64_THIRD_PARTY_DIR}gtest/lib/
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
