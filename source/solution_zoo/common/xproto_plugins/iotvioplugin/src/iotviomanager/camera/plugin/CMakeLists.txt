cmake_minimum_required(VERSION 2.8)
option(PARENT_BUILD "is build from parent" OFF)

message("sensor plugin PARENT_BUILD:${PARENT_BUILD}")
if (NOT ${PARENT_BUILD})
  set(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
endif ()

project(sensor_plugin)

if (NOT ${PARENT_BUILD})
  option(RELEASE_LIB "build version of release" ON)
  message("config types: ${CMAKE_CONFIGURATION_TYPES}")
  add_definitions(-DHR_POSIX)
  add_definitions(-DHR_LINUX)

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
  endif()

  list(APPEND CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 ")
  string(REGEX REPLACE ";" " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
  string(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(MY_OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/output)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/deps

)
else ()
  set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME})
endif ()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${MY_OUTPUT_ROOT}/include)

add_subdirectory(src/f37_plugin)
add_subdirectory(src/gc4c33_plugin)

