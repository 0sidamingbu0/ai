cmake_minimum_required(VERSION 2.8)
project(MultitaskPostProcessMethod)

include_directories(
    include
)

link_directories(${CMAKE_SOURCE_DIR}/lib/)

set(SOURCE_FILES
        src/MultitaskPostProcessMethod.cpp
        src/result_util.cpp
)

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

set(DYNAMIC_DEPS_LIBS bpu_predict cnn_intf hbrt_bernoulli_aarch64 cam vio)
target_link_libraries(${PROJECT_NAME} ${DYNAMIC_DEPS_LIBS} DnnPostProcessMethod)

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)

install(FILES ${PROJECT_SOURCE_DIR}/include/MultitaskPostProcessMethod/MultitaskPostProcessMethod.h
        DESTINATION ${MY_OUTPUT_ROOT}/include/MultitaskPostProcessMethod)

install(FILES ${PROJECT_SOURCE_DIR}/include/MultitaskPostProcessMethod/result_util.h
        DESTINATION ${MY_OUTPUT_ROOT}/include/MultitaskPostProcessMethod)

install(TARGETS ${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/lib)
