cmake_minimum_required(VERSION 2.8)

# 项目名称
project(alsadevice)

# 添加头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

file(GLOB PROJECT_INC
    "include/*.h"
)

file(GLOB PROJECT_SRC
    "src/*.cpp"
)

set(SOURCE_FILES ${PROJECT_SRC} ${PROJECT_INC})

set(BASE_LIB asound)
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${BASE_LIB})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${MY_OUTPUT_ROOT}/include/)
install(TARGETS ${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/lib)
