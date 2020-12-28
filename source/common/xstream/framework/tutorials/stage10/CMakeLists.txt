cmake_minimum_required(VERSION 2.8)

# 设置头文件搜索路径
include_directories(./method)

set(BASE_LIBRARIES xstream bpu_predict hobotlog opencv_world jsoncpp pthread rt
    DnnPredictMethod DnnPostProcessMethod xstream-imagetools
# hbrt_bernoulli_aarch64 cnn_intf
)

add_library(Mobilenetv2PostProcessMethod STATIC 
    method/Mobilenetv2PostProcessMethod/mobilenetv2_post_process_method.cc)
target_link_libraries(Mobilenetv2PostProcessMethod ${BASE_LIBRARIES})

add_library(Mobilenetv2PredictMethod STATIC 
    method/Mobilenetv2PredictMethod/mobilenetv2_predict_method.cc)
target_link_libraries(Mobilenetv2PredictMethod ${BASE_LIBRARIES})

add_library(Yolov3PostProcessMethod STATIC 
    method/Yolov3PostProcessMethod/yolov3_post_process_method.cc)
target_link_libraries(Yolov3PostProcessMethod ${BASE_LIBRARIES})

add_library(Yolov3PredictMethod STATIC 
    method/Yolov3PredictMethod/yolov3_predict_method.cc)
target_link_libraries(Yolov3PredictMethod ${BASE_LIBRARIES})


set(DEP_LIBRARIES viowrapper)

add_executable(stage10_example
    main.cc
    method/method_factory.cc)

target_link_libraries(stage10_example
    Mobilenetv2PostProcessMethod
    Mobilenetv2PredictMethod
    Yolov3PostProcessMethod
    Yolov3PredictMethod
    ${BASE_LIBRARIES}
    ${DEP_LIBRARIES})


install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/method/Mobilenetv2PostProcessMethod/mobilenetv2_post_process_method.h
    DESTINATION ${OUTPUT_ROOT}/Mobilenetv2PostProcessMethod/include)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/method/Mobilenetv2PredictMethod/mobilenetv2_predict_method.h
    DESTINATION ${OUTPUT_ROOT}/Mobilenetv2PredictMethod/include)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/method/Yolov3PostProcessMethod/yolov3_post_process_method.h
    DESTINATION ${OUTPUT_ROOT}/Yolov3PostProcessMethod/include)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/method/Yolov3PredictMethod/yolov3_predict_method.h
    DESTINATION ${OUTPUT_ROOT}/Yolov3PredictMethod/include)

install(TARGETS Mobilenetv2PostProcessMethod
        DESTINATION ${OUTPUT_ROOT}/Mobilenetv2PostProcessMethod/lib)
install(TARGETS Mobilenetv2PredictMethod
        DESTINATION ${OUTPUT_ROOT}/Mobilenetv2PredictMethod/lib)
install(TARGETS Yolov3PostProcessMethod
        DESTINATION ${OUTPUT_ROOT}/Yolov3PostProcessMethod/lib)
install(TARGETS Yolov3PredictMethod
        DESTINATION ${OUTPUT_ROOT}/Yolov3PredictMethod/lib)