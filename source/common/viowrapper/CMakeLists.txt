cmake_minimum_required(VERSION 2.8)

option(ALONE_BUILD "is build from parent" OFF)
option(PLATFORM_X2 "x2 platform" ON)
option(PLATFORM_X3 "x3 platform" OFF)
option(IOT_VIO_API "iot vio api" OFF)

if(${ALONE_BUILD})
include(cmake/hobot_util.cmake)
endif()
project(vio_wrapper)
set(CMAKE_CXX_STANDARD 11)
set(PLATFORM_LINUX TRUE)
if (${PLATFORM_MAC})
  add_definitions(-DHR_POSIX)
  add_definitions(-DHR_MAC)
elseif (${PLATFORM_LINUX})
  add_definitions(-DHR_POSIX)
  add_definitions(-DHR_LINUX)
  set(PLATFORM_NAME linux-x64)
elseif (${PLATFORM_ANDROID})
  set(ENABLE_NEON "true")
  add_definitions(-DENABLE_OMP)
  add_definitions(-DHR_POSIX)
  add_definitions(-DHR_ANDROID)
elseif (${PLATFORM_WIN})
  set(CMAKE_CXX_FLAGS_RELEASE "/MD")
  set(CMAKE_CXX_FLAGS_DEBUG "/MDd")
  add_definitions(-DHR_WIN)
  add_definitions(-D_MBCS)
  add_definitions(-DHOBOT_EXPORTS)
else ()
  message(FATAL_ERROR "invoke cmake with -DPLATFORM_MAC=TRUE|-DPLATFORM_LINUX=TRUE|-DPLATFORM_ANDROID=TRUE|-DPLATFORM_WIN=TRUE")
  return()
endif()

if (${PLATFORM_X2})
  add_definitions(-DX2)
elseif (${PLATFORM_X3})
  add_definitions(-DX3)
  if (${IOT_VIO_API})
    add_definitions(-DX3_IOT_VIO)
  else ()
    add_definitions(-DX3_X2_VIO)
  endif ()
else ()
  add_definitions(-DX2)
endif()

set(CMAKE_BUILD_TYPE Release)

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEBUG -g -O0 -fPIC ")
endif()

message("config types: ${CMAKE_CONFIGURATION_TYPES}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include/
)

set(OpenCVLib opencv_world xstream-imagetools libturbojpeg.a libyuv.a)
set(BaseLib vio cam pthread jsoncpp hobotlog dl rt)

if (${PLATFORM_X2})
  set(X2_PREBUILT_LIB libcam.so libvio.so)
  set(SOURCE_FILES src/vio_wrapper.cpp)
elseif (${PLATFORM_X3})
  if (${IOT_VIO_API})
    # if (NOT (${IOT_VIO_SRC}))
    #   set (AI_EXPRESS_ROOT ${PROJECT_SOURCE_DIR}/../..)
    #   set (IOT_VIO_SRC ${AI_EXPRESS_ROOT}/xsdk/common/xproto/plugins/iotvioplugin)
    #   message(STATUS "ai express root path = ${AI_EXPRESS_ROOT}")
    #   message(STATUS "iot vio src path = ${IOT_VIO_SRC}")
    #   include_directories(
    #     ${AI_EXPRESS_ROOT}/x3_prebuilt/include
    #     ${AI_EXPRESS_ROOT}/x3_prebuilt/include/hbmedia
    #     ${AI_EXPRESS_ROOT}/xsdk/common/xproto/framework/include
    #     ${AI_EXPRESS_ROOT}/xsdk/common/xproto/msgtype/include
    #     ${AI_EXPRESS_ROOT}/xsdk/common/xstream/vision_type/include
    #     ${AI_EXPRESS_ROOT}/xsdk/common/xstream/imagetools/include
    #     ${IOT_VIO_SRC}/include
    #     )
    #   link_directories(
    #     ${AI_EXPRESS_ROOT}/x3_prebuilt/lib
    #     ${AI_EXPRESS_ROOT}/x3_prebuilt/lib/hbmedia
    #     )
      # file(GLOB PROJECT_SRC
      #   "${AI_EXPRESS_ROOT}/xsdk/common/xstream/imagetools/src/*.cpp"
      #   )
      # set(OpenCVLib opencv_world libturbojpeg.a libyuv.a)
    # endif()

    set(X3_HAPI_LIB isp iar isp_algo tinyalsa multimedia z avformat hbmedia)
    file(GLOB PROJECT_SRC
      "${IOT_VIO_SRC}/src/iotviomanager/*.cpp"
      "${IOT_VIO_SRC}/src/iotviomanager/camera/*.cpp"
      )
    file(GLOB PROJECT_INC
      "${IOT_VIO_SRC}/include/iotviomanager/*.h"
      "${PROJECT_SOURCE_DIR}/include/*.h"
      )
    set(SOURCE_FILES
      src/vio_wrapper_global.cpp
      ${PROJECT_SRC}
      ${PROJECT_INC}
      )
  else ()
    set(X2_PREBUILT_LIB libcam.so libvio.so)
    set(SOURCE_FILES src/vio_wrapper_global.cpp)
  endif()
endif()

add_library(viowrapper STATIC ${SOURCE_FILES})
if (${PLATFORM_X3})
  target_link_libraries(viowrapper ${OpenCVLib} ${X3_HAPI_LIB} ${BaseLib})
else ()
  target_link_libraries(viowrapper ${OpenCVLib} ${X2_PREBUILT_LIB} ${BaseLib})
endif ()

if(${ALONE_BUILD})
  set(OUTPUT_ROOT ${CMAKE_SOURCE_DIR}/output)
  set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME})
else()
  if (${PLATFORM_X3})
    set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/x3)
  else ()
    set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/x2)
  endif ()
endif()


# if(${ALONE_BUILD})
# add_subdirectory(test)
# endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
   DESTINATION ${MY_OUTPUT_ROOT}/)
#install(DIRECTORY ${PROJECT_SOURCE_DIR}/lib
#   DESTINATION ${MY_OUTPUT_ROOT}/)
install(TARGETS viowrapper
        DESTINATION ${MY_OUTPUT_ROOT}/lib)
