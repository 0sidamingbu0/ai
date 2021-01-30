cmake_minimum_required(VERSION 2.8)
project(multisourceinput)

set(SOURCE_FILES
        main.cpp
        method_factory.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(BASE_LIBRARIES vioplugin commongdcplugin multisourcesmartplugin multisourcewebsocketplugin
        PredictMethod PostProcessMethod
        FasterRCNNMethod CNNMethod bpu_predict MergeMethod SnapShotMethod GradingMethod vote_method
        MOTMethod ipc_tracking xstream xstream-imagetools hobotlog
        jsoncpp opencv_world)

target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES})

set(MY_OUTPUT_ROOT ${OUTPUT_ROOT}/${PROJECT_NAME}/)
install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/configs/
        DESTINATION ${MY_OUTPUT_ROOT}/configs/)
install(TARGETS ${PROJECT_NAME}
        DESTINATION ${MY_OUTPUT_ROOT}/)
