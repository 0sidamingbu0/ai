cmake_minimum_required(VERSION 2.8)
project(yolov3_solution)

set(SOURCE_FILES
        main.cpp
        method_factory.cpp
        )

include_directories(
    ExampleSmartPlugin/include
    ExampleWebsocketPlugin/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/xstream/framework/tutorials/stage10/method/Mobilenetv2PostProcessMethod
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/xstream/framework/tutorials/stage10/method/Mobilenetv2PredictMethod
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/xstream/framework/tutorials/stage10/method/Yolov3PostProcessMethod
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/xstream/framework/tutorials/stage10/method/Yolov3PredictMethod
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(BASE_LIBRARIES vioplugin
        DnnPredictMethod DnnPostProcessMethod
        bpu_predict xstream
        websocketplugin
        xstream-imagetools hobotlog vision_type_util jsoncpp opencv_world examplesmartplugin
        examplewebsocketplugin Mobilenetv2PredictMethod Mobilenetv2PostProcessMethod
        Yolov3PostProcessMethod Yolov3PredictMethod)

target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)
install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/configs/
        DESTINATION ${MY_OUTPUT_ROOT}/configs/)
install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/models/
        DESTINATION ${MY_OUTPUT_ROOT}/models/)
install(TARGETS ${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/)