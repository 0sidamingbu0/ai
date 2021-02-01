cmake_minimum_required(VERSION 2.8)

project(LowPassFilterMethod)

#option(SUPPORT_RGB_NIR_MERGE "support rgb nir merge" OFF)
#if (${SUPPORT_RGB_NIR_MERGE})
#    add_definitions(-DRGB_NIR_MERGE)
#endif ()

include_directories(
        include
#        include/MergeMethod
)

set(SOURCE_FILES
        src/LowPassFilterMethod/LowPassFilterMethod.cpp
        src/LowPassFilterMethod/data_type.cpp
        )

add_library(LowPassFilterMethod STATIC ${SOURCE_FILES})

set(BASE_LIBRARIES xstream hobotlog jsoncpp opencv_world)

target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
        DESTINATION ${MY_OUTPUT_ROOT})

install(TARGETS ${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/lib)

#add_subdirectory(test)
#add_subdirectory(example)
