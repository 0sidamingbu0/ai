cmake_minimum_required(VERSION 2.8)
project(xproto_msgtype)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/xproto_msgtype/protobuf
)
# 添加链接库路径
link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/build/lib
)
# 头文件路径信息
file(GLOB PROJECT_INC
        "include/*.h"
        "include/xproto_msgtype/*.h"
        "include/xproto_msgtype/protobuf/*.h"
        )

# 源文件路径信息
file(GLOB PROJECT_SRC
        "src/*.cc"
        "src/protobuf/*.cc"
        )

set(SOURCE_FILES
        ${PROJECT_SRC}
        ${PROJECT_INC}
        )

set(BASE_LIBRARIES ${BASE_LIBRARIES} hobotlog vision_type_util protobuf ${ARCH_LIBS})

set(OpenCVLib opencv_world)
set(BASE_LIBRARIES ${BASE_LIBRARIES} ${OpenCVLib})
add_library(xproto_msgtype STATIC ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} xproto ${BASE_LIBRARIES})
# add_subdirectory(test)

# make install才会安装
if (NOT ${PARENT_BUILD})
set(OUTPUT_ROOT ${CMAKE_SOURCE_DIR}/output)
endif ()
set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME})
install(DIRECTORY
        ${CMAKE_SOURCE_DIR}/build/lib/
        DESTINATION ${MY_OUTPUT_ROOT}/lib/)
install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/include)
