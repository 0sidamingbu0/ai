#!/usr/bin/sh

function usage() {
    echo "usage: sh run_ut.sh"
    exit 1
}

function CheckRet() {
if [ $? -ne 0 ] ; then
    echo "failed to "$1
    exit 1
fi
}

function RunXProto(){
  ${UT_DIR}/xproto/xproto_unit_test 
  # ${UT_DIR}/xproto/vioplugin_test
  # rm configs -rf
}

function RunXProtoPlugin(){
  ${UT_DIR}/xproto_plugin/smartplugin_test 
}

function RunVisionType(){
  ${UT_DIR}/vision_type/gtest_vision_type
}

function RunXStream(){
  cd xstream
  ./xstream_test
  #${UT_DIR}/xstream/xstream_multisource_test
  ./xstream_threadmodel_test
  ./xstream_threadorder_test
  ./xstream_threadsafe_test
  ./xstream_unit_test
  ./config_test
  ./cpp_api_test
  ./node_test
  ./workflow_test
  # ./timer_test
  # ./method_init_test
  # ./disable_method_test
  # ./profiler_test
  # rm test -rf
  # rm config -rf
  cd ../
}

function RunCNNMethod(){
  cd cnn_method
  export LD_LIBRARY_PATH=../../lib/
  ./CNNMethod_unit_test
  # rm config -rf
  # rm data -rf
  cd ..
}

function RunFasterrcnnMethod(){
 ${UT_DIR}/fasterrcnn_method/gtest_fasterrcnn
}

function RunGrading(){
  cd grading_method
  ./gtest_grading
  ./grading_example
  cd ..
}

function RunMerge(){
  cd merge_method
  export LD_LIBRARY_PATH=../../lib/
  ./gtest_head_face
  cd ..
}

function RunMot(){
  cd mot_method
  ./gtest_mot
  cd ..
}

function RunSnapshot(){
  cd snapshot_method
  export LD_LIBRARY_PATH=../../lib/
  ./gtest_snapshot
  cd ..
}

function RunVoteMehtod(){
  ${UT_DIR}/vote_method/vote_method_test
}

function ChangeRunMode(){
  sed -i "s#\(\./face_solution/configs/face_solution.json ./configs/visualplugin_face.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./face_solution/configs/face_recog_solution.json ./configs/visualplugin_face.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./body_solution/configs/body_solution.json ./configs/visualplugin_body.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./body_solution/configs/xbox_solution.json ./configs/visualplugin_body.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./body_solution/configs/behavior_solution.json ./configs/visualplugin_body.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./body_solution/configs/gesture_solution.json ./configs/visualplugin_body.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./video_box/configs/body_solution.json  ./configs/visualplugin_body.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\./body_solution/configs/dance_solution.json ./configs/visualplugin_body.json -i\).*#\1 ${1}#g" run.sh
  sed -i "s#\(\${face_body_multisource_vio} ./face_body_multisource/configs/face_body_solution.json -i\).*#\1 ${1}#g" run.sh
}

function stop_nginx(){
  nginx_flag=$(ps | grep nginx | grep -v "grep")
  if [ -n "${nginx_flag}" ]; then
    killall -9 nginx
  fi  
}

function RunSolutions(){
  cd ..
 
  ARCHITECTURE=${1}
  if [ ${ARCHITECTURE} == "x3" ];then
     ChangeRunMode ut
    # log_level i:info w:warning
    log_level=w
    # board_type 3:x3sdb 4:x3dvb
    board_type=3
    # # fb_resolution 1:1080p feedback  2:2160p feedback
    # fb_resolution=1
    # face x3dev hg cache
    # sh run.sh ${log_level} ut 1 ${board_type} 2 ${fb_resolution} 1
    sh run.sh ${log_level} ut 1 ${board_type} 2 2
    # # face x3dev hg jpg
    # sh run.sh ${log_level} ut 1 ${board_type} 2 2
    # # face x3dev hg nv12
    # sh run.sh ${log_level} ut 1 ${board_type} 2 3
    # # face_recog x3dev hg cache
    # sh run.sh ${log_level} ut 2 ${board_type} 2 1
    # # face_recog x3dev hg jpg
    # sh run.sh ${log_level} ut 2 ${board_type} 2 2
    # # face_recog x3dev hg nv12
    # sh run.sh ${log_level} ut 2 ${board_type} 2 3
    # body x3dev hg cache
    # sh run.sh ${log_level} ut 3 ${board_type} 2 ${fb_resolution} 1
    sh run.sh ${log_level} ut 3 ${board_type} 2 3
    # # body x3dev hg jpg
    # sh run.sh ${log_level} ut 3 ${board_type} 2 2
    # # body x3dev hg nv12
    # sh run.sh ${log_level} ut 3 ${board_type} 2 3
    # # xbox x3dev hg cache
    # sh run.sh ${log_level} ut 4 ${board_type} 2 1
    # # xbox x3dev hg jpg
    # sh run.sh ${log_level} ut 4 ${board_type} 2 2
    # # xbox x3dev hg nv12
    # sh run.sh ${log_level} ut 4 ${board_type} 2 3
    # # behavior x3dev hg cache
    # sh run.sh ${log_level} ut 5 ${board_type} 2 1
    # # behavior x3dev hg jpg
    # sh run.sh ${log_level} ut 5 ${board_type} 2 2
    # # behavior x3dev hg nv12
    # sh run.sh ${log_level} ut 5 ${board_type} 2 3
    # # gesture x3dev hg cache
    # sh run.sh ${log_level} ut 6 ${board_type} 2 1
    # # gesture x3dev hg jpg
    # sh run.sh ${log_level} ut 6 ${board_type} 2 2
    # # gesture x3dev hg nv12
    # sh run.sh ${log_level} ut 6 ${board_type} 2 3
    # video_box x3dev hg cache
    #sh run.sh ${log_level} ut 7 ${board_type} 2 1
    # # video_box x3dev hg jpg
    # sh run.sh ${log_level} ut 7 ${board_type} 2 2
    # # video_box x3dev hg nv12
    # sh run.sh ${log_level} ut 7 ${board_type} 2 3
    # # tv_uvc x3dev hg cache
    # sh run.sh ${log_level} ut 8 ${board_type} 2 1
    # # tv_uvc x3dev hg jpg
    # sh run.sh ${log_level} ut 8 ${board_type} 2 2
    # # tv_uvc x3dev hg nv12
    # sh run.sh ${log_level} ut 8 ${board_type} 2 3
    # face_body_multisource x3dev hg cache
    sh run.sh ${log_level} ut 9 ${board_type} 2 1
    # face_body_multisource x3dev hg jpg
    sh run.sh ${log_level} ut 9 ${board_type} 2 2
    # face_body_multisource x3dev hg nv12
    sh run.sh ${log_level} ut 9 ${board_type} 2 3
    ChangeRunMode normal
  elif [ ${ARCHITECTURE} == "x2" ];then
    ChangeRunMode ut
    sh run.sh face 96board
    sh run.sh face_recog 96board
    sh run.sh body 96board
    sh run.sh xbox 96board
    sh run.sh behavior 96board
    # sh run.sh video_box 96board  # No support plan for the x2 platform
    sh run.sh gesture 96board
    sh run.sh tv_dance 96board
    sh run.sh face_body_multisource 96board
    # sh run.sh face 96board hg cache # Cannot run
    # sh run.sh face 96board hg jpg  # Cannot run
    # sh run.sh face 96board hg nv12 # Cannot run
    ChangeRunMode normal
  fi
  stop_nginx
  cd unit_test
}

function RunSSDMethod(){
  cd ssd_method
  export LD_LIBRARY_PATH=../../lib/
  ./ssd_method_test
  cd ..
}

# ARCHITECTURE="x2"
# if [ $# -ge 2 ]
# then
#   if [ ${1} == "x2" ]
#   then
#     ARCHITECTURE="x2"
#   elif [ ${1} == "x3" ]
#   then
#     ARCHITECTURE="x3"
#   else
#     usage
#   fi
# else
#   usage
# fi

# VIOINTERFACE="VIO_HAPI"
# if [ $# -ge 3 ]
# then
#   if [ ${3} == "VIO_NORMAL" ]
#   then
#     VIOINTERFACE="VIO_NORMAL"
#   elif [ ${3} == "VIO_HAPI" ]
#   then
#     VIOINTERFACE="VIO_HAPI"
#   else
#     usage
#   fi
# fi

#########test succ begin#################
set -eux

export LD_LIBRARY_PATH=../lib/:../../lib
UT_DIR=.
ARCHITECTURE=$(cat platform.tmp)

RunXProto
RunXProtoPlugin
RunVisionType
RunXStream
RunGrading
RunSnapshot
RunMerge
RunMot
RunSSDMethod ${ARCHITECTURE}

RunSolutions ${ARCHITECTURE}
#########test succ end##################

# RunCNNMethod ${ARCHITECTURE}
# RunFasterrcnnMethod ${ARCHITECTURE}
# RunVoteMehtod



