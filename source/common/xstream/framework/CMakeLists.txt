cmake_minimum_required(VERSION 2.8)
project(xstream
	LANGUAGES CXX
)

# PARENT_BUILD:OFF buid xstream separately
# PARENT_BUILD:ON git-repo build all
option(PARENT_BUILD "is build from parent" OFF)
if (NOT ${PARENT_BUILD})
	get_filename_component(XSTREAM_CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	# message("xstream current dir init:${XSTREAM_CURRENT_DIR}")
	set(RELEASE_ROOT_DIR)
	function(get_parent_dir arg dir n)
		foreach(i RANGE ${ARGV2})
			get_filename_component(dir ${dir} DIRECTORY)
			#message("i:${i} dir:${dir}")
		endforeach(i)
		set(RELEASE_ROOT_DIR ${dir} PARENT_SCOPE)
	endfunction()

	get_parent_dir(XSTREAM_CURRENT_DIR ${XSTREAM_CURRENT_DIR} 2)
	# message("xstream current dir:${XSTREAM_CURRENT_DIR}")
	# message("source dir:${RELEASE_ROOT_DIR}")

	option(X86_ARCH "x86 architecture" OFF)
	option(X86_CENTOS "centos release" OFF)

	# build separately install directory
  set(OUTPUT_ROOT ${CMAKE_SOURCE_DIR}/output)
endif()

option(XSTREAM_HOBOTSDK_ENGINE "choose hobotsdk engine" OFF)
option(RELEASE_LIB "build version of release" OFF)

option(C_INTERFACE "add c interface" OFF)
if(C_INTERFACE)
	add_definitions(-DC_INTERFACE)
endif()

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
	endif()

	# x86 architecture
	if (${X86_ARCH})
		if (${X86_CENTOS})
			message("tool chain:${CMAKE_CXX_COMPILER}")
			add_definitions(-std=c++11)
			include_directories(
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/centos/jsoncpp/include/
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/centos/hobotlog/include/
				${XSTREAM_CURRENT_DIR}/vision_type/include/
			)
			link_directories(
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/centos/jsoncpp/lib
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/centos/hobotlog/lib
			)
		else ()
			message("tool chain:${CMAKE_CXX_COMPILER}")
			add_definitions(-std=c++11)
			include_directories(
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/ubuntu/jsoncpp/include/
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/ubuntu/hobotlog/include/
				${XSTREAM_CURRENT_DIR}/vision_type/include/
			)
			link_directories(
				${XSTREAM_CURRENT_DIR}/framework/third_party/x86/ubuntu/hobotlog/lib
			)
		endif ()
	else ()
		set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
		message("tool chain:${CMAKE_CXX_COMPILER}")
		include_directories(
			${RELEASE_ROOT_DIR}/deps/jsoncpp/include/
			${RELEASE_ROOT_DIR}/deps/hobotlog/include/
			${XSTREAM_CURRENT_DIR}/vision_type/include/
			${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/jsoncpp/include/
			${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/hobotlog/include/
			${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/opencv/include/
			${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/gtest/include/
		)
		link_directories(
				${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/hobotlog/lib
				${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/jsoncpp/lib
				${XSTREAM_CURRENT_DIR}/framework/third_party/aarch64/gtest/lib
			)
	endif ()
endif ()

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include/common
	${CMAKE_CURRENT_SOURCE_DIR}/include/log
	${CMAKE_CURRENT_SOURCE_DIR}/include/timer
)

include_directories(SYSTEM /usr/include/python2.7)

set(SOURCE_FILES
	src/common/com_func.cpp
	src/profiler.cpp
	src/xstream.cpp
	src/xstream_data.cpp
	src/xstream_config.cpp
	src/method.cpp
)
if(C_INTERFACE)
	set(SOURCE_FILES ${SOURCE_FILES} src/xstream_capi.cpp src/xstream_capi_type.cpp src/xstream_c_api.cpp)
endif()

if(XSTREAM_HOBOTSDK_ENGINE)
	add_definitions(-DHOBOTSDK_ENGINE)
	include_directories(include/hobotxstream/hobotsdk)
	set(HOBOTSDK_ENGINE_SOURCES
		include/hobotxstream/hobotsdk/BaseMessage.h
		include/hobotxstream/hobotsdk/NodeV2.h
		include/hobotxstream/hobotsdk/schedulerV2.h
		include/hobotxstream/hobotsdk/ThreadSafeMethod.h
		include/hobotxstream/hobotsdk/base_modules/CollectModule.h
		include/hobotxstream/hobotsdk/base_modules/DefaultDispatchModule.h
		include/hobotxstream/hobotsdk/base_modules/DispatchModule.h
		include/hobotxstream/hobotsdk/base_modules/MethodModule.h
		include/hobotxstream/hobotsdk/base_modules/ReorderDispatchModule.h
		src/hobotsdk/NodeV2.cpp
		src/hobotsdk/schedulerV2.cpp
		src/hobotsdk/base_modules/CollectModule.cpp
		src/hobotsdk/base_modules/DefaultDispatchModule.cpp
		src/hobotsdk/base_modules/DispatchModule.cpp
		src/hobotsdk/base_modules/MethodModule.cpp
		src/hobotsdk/base_modules/ReorderDispatchModule.cpp
	)
	set(SOURCE_FILES ${SOURCE_FILES} ${HOBOTSDK_ENGINE_SOURCES})
else()
	set(NATIVE_ENGINE_SOURCES
		src/method_manager.cpp
		src/timer/timer.cpp
		src/node.cpp
		src/scheduler.cpp
		src/xthread.cpp
		src/thread_manager.cpp
		src/xthread_pool.cpp
	)
	set(SOURCE_FILES ${SOURCE_FILES} ${NATIVE_ENGINE_SOURCES})
endif()


option(BUILD_SHARED_LIBS "build shared libs" OFF)
add_library(xstream ${SOURCE_FILES})

target_link_libraries(xstream
	PUBLIC rt)

if(XSTREAM_HOBOTSDK_ENGINE)
    target_link_libraries(xstream
			PUBLIC hobotsdk hobotthread jsoncpp hobotlog)
else()
    target_link_libraries(xstream
                          PUBLIC jsoncpp)
endif()

add_subdirectory(example/bbox_filter)
add_subdirectory(test)
add_subdirectory(benchmark)

add_subdirectory(tutorials/stage1)
add_subdirectory(tutorials/stage2)
add_subdirectory(tutorials/stage3)
add_subdirectory(tutorials/stage4)
add_subdirectory(tutorials/stage5)
add_subdirectory(tutorials/stage6)
add_subdirectory(tutorials/stage7)
add_subdirectory(tutorials/stage8)
add_subdirectory(tutorials/stage9)

if (${BUILD_DNNMETHOD})
  add_subdirectory(methods/DnnPredictMethod)
  add_subdirectory(methods/DnnPostProcessMethod)
  add_subdirectory(tutorials/stage10)
endif()

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME})

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxsdk/version.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxsdk/xstream_data.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxsdk/xstream_error.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxsdk/xstream_sdk.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/hobotxsdk
)
install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxstream/json_key.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxstream/method_factory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxstream/method.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxstream/profiler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobotxstream/xstream_config.h
  DESTINATION ${MY_OUTPUT_ROOT}/include/hobotxstream
)

install(TARGETS ${PROJECT_NAME}
	DESTINATION ${MY_OUTPUT_ROOT}/lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
	DESTINATION ${MY_OUTPUT_ROOT}/
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tutorials
	DESTINATION ${MY_OUTPUT_ROOT}/
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party
	DESTINATION ${MY_OUTPUT_ROOT}/tutorials
)
