#!/bin/sh
usage() {
   echo "usage: sh run.sh "
   exit 1
}

chmod +x start_nginx.sh
sh start_nginx.sh

if [ ! -L "./lib/libgomp.so.1" ];then
  ln -s /lib/libgomp.so ./lib/libgomp.so.1
  echo "create symbolic link in ./lib directory"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/
vio_cfg_file=./configs/vio_config.json.96board
vio_pipe_path="configs/vio/x3dev/"
vio_pipe_pre="iot_vio_x3_"
fb_pic_path="configs/vio_hg/"
fb_nv12_list="configs/vio_hg/name_nv12.list"
fb_jpg_list="configs/vio_hg/name_jpg.list"
sensor1="imx327"
sensor2="os8a10"
vio_cfg_name="./configs/vio_config.json"
vio_mode="panel_camera"
cam_mode="panel_camera"

#default solution mode is body in ut mode
solution_mode="body"
solution_mode_aNum=3
#default platform is x3dev in ut mode
platform="x3dev"
platform_aNum=4
#default vio type is fb in ut mode
vio_type="fb"
vio_type_aNum=2
#default fb_mode is cached_image_list in ut mode
fb_mode_aNum=1
fb_mode="cached_image_list"
#default sensor is os8a10 in ut mode
sensor="os8a10"
sensor_aNum=2
vio_pipe_file="${vio_pipe_path}${vio_pipe_pre}${sensor}.json"
#default fb_res is 1080p
fb_res=1080_fb
fb_res_aNum=1

if [ $# -gt 1 ]
then
  if [ $2 == "ut" ]; then
    run_mode="ut"
  elif [ $2 == "cmd_normal" ]; then
    run_mode="cmd_normal"
  elif [ $2 == "ruku" ]; then
    run_mode="ruku"
  elif [ $2 == "db_list" ]; then
    run_mode="db_list"
  elif [ $2 == "db_drop_table" ]; then
    run_mode="db_drop_table"
  else
    usage
  fi
  echo "enter vio solution ${run_mode} mode!!!"
  if [ -n "$3" ];then
    solution_mode_aNum=$3
    echo "solution_mode_aNum is $solution_mode_aNum"
  fi
  if [ -n "$4" ];then
    platform_aNum=$4
    echo "platform_aNum is $platform_aNum"
  fi
  if [ -n "$5" ];then
    vio_type_aNum=$5
    echo "vio_type_aNum is $vio_type_aNum"
  fi
  if [ $vio_type == "cam" ];then
    if [ -n "$6" ];then
      sensor_aNum=$6
      echo "sensor_aNum is $sensor_aNum"
    fi
  elif [ $vio_type == "fb" ];then
    if [ -n "$6" ];then
      fb_mode_aNum=$6
      echo "fb_mode_aNum is $fb_mode_aNum"
    fi
  else
    echo "not support vio_type:$vio_type in ut mode"
    exit 1
  fi
fi

function stop_nginx(){
  nginx_flag=$(ps | grep nginx | grep -v "grep")
  if [ -n "${nginx_flag}" ]; then
    killall -9 nginx
  fi
}

visual_cfg_func() {
  layer=$1
  width=$2
  height=$3
  layer_720p=$4
  layer_1080p=$5
  layer_2160p=$6
  sed -i 's#\("layer": \).*#\1'$layer',#g' configs/visualplugin_face.json
  sed -i 's#\("layer": \).*#\1'$layer',#g' configs/visualplugin_body.json
  sed -i 's#\("layer": \).*#\1'$layer',#g' configs/visualplugin_vehicle.json
  sed -i 's#\("layer": \).*#\1'$layer',#g' configs/visualplugin_multisource.json
  sed -i 's#\("image_width": \).*#\1'$width',#g' configs/visualplugin_face.json
  sed -i 's#\("image_width": \).*#\1'$width',#g' configs/visualplugin_body.json
  sed -i 's#\("image_width": \).*#\1'$width',#g' configs/visualplugin_vehicle.json
  sed -i 's#\("image_width": \).*#\1'$width',#g' configs/visualplugin_multisource.json
  sed -i 's#\("image_height": \).*#\1'$height',#g' configs/visualplugin_face.json
  sed -i 's#\("image_height": \).*#\1'$height',#g' configs/visualplugin_body.json
  sed -i 's#\("image_height": \).*#\1'$height',#g' configs/visualplugin_vehicle.json
  sed -i 's#\("image_height": \).*#\1'$height',#g' configs/visualplugin_multisource.json
  sed -i 's#\("720p_layer": \).*#\1'$layer_720p',#g' configs/visualplugin_body.json
  sed -i 's#\("1080p_layer": \).*#\1'$layer_1080p',#g' configs/visualplugin_body.json
  sed -i 's#\("2160p_layer": \).*#\1'$layer_2160p',#g' configs/visualplugin_body.json
}

solution_cfg_func() {
  resolution=$1
  cp -rf body_solution/configs/box_filter_config_${resolution}M.json  body_solution/configs/box_filter_config.json
  cp -rf body_solution/configs/gesture_multitask_${resolution}M.json  body_solution/configs/gesture_multitask.json
  cp -rf body_solution/configs/multitask_config_${resolution}M.json  body_solution/configs/multitask_config.json
  cp -rf body_solution/configs/multitask_with_hand_${resolution}M.json  body_solution/configs/multitask_with_hand.json
  cp -rf body_solution/configs/multitask_with_hand_960x544_${resolution}M.json  body_solution/configs/multitask_with_hand_960x544.json
  cp -rf body_solution/configs/segmentation_multitask_${resolution}M.json  body_solution/configs/segmentation_multitask.json
  cp -rf face_solution/configs/filter_config_${resolution}M.json face_solution/configs/filter_config.json
  cp -rf face_solution/configs/face_pose_lmk_${resolution}M.json face_solution/configs/face_pose_lmk.json
  cp -rf face_solution/configs/predict_face_pose_lmk_${resolution}M.json face_solution/configs/predict_face_pose_lmk.json
  cp -rf vehicle_solution/configs/vehicle_multitask_${resolution}M.json vehicle_solution/configs/vehicle_multitask.json
  cp -rf vehicle_solution/configs/config_match_${resolution}M.json vehicle_solution/configs/config_match.json
  cp -rf yolov3_solution/configs/yolov3_predict_method_${resolution}M.json yolov3_solution/configs/yolov3_predict_method.json
  cp -rf yolov3_solution/configs/yolov3_post_process_method_${resolution}M.json yolov3_solution/configs/yolov3_post_process_method.json
}

set_fb_cam_data_source_func() {
  vio_cfg_file=$1
  fb_mode=$2
  cam_mode=$3
  data_source_num=$4

  for i in $(seq 1 $data_source_num); do
    if [ $i == 1 ];then
      vio_mode=$fb_mode
    elif [ $i == 2 ];then
      vio_mode=$cam_mode
    fi
    data_src_line=`cat -n ${vio_cfg_file} | grep -w "data_source" | awk '{print $'$i'}' | sed -n ''$i'p'`
    sed -i ''${data_src_line}'s#\("data_source": \).*#\1"'${vio_mode}'",#g' ${vio_cfg_file}
  done
}

set_data_source_func() {
  vio_cfg_file=$1
  vio_mode=$2
  data_source_num=$3

  for i in $(seq 1 $data_source_num); do
    data_src_line=`cat -n ${vio_cfg_file} | grep -w "data_source" | awk '{print $1}' | sed -n ''$i'p'`
    #echo "i:$i vio_cfg_file:$vio_cfg_file data_src_line:$data_src_line vio_mode:$vio_mode"
    sed -i ''${data_src_line}'s#\("data_source": \).*#\1"'${vio_mode}'",#g' ${vio_cfg_file}
  done
}

set_cam_pipe_file_func() {
  vio_mode=$1
  sensor=$2
  vio_pipe_file=${vio_pipe_path}${vio_pipe_pre}${sensor}.json
  echo "vio_mode: $vio_mode"
  echo "vio_pipe_file: $vio_pipe_file"
  sed -i 's#\("'${vio_mode}'": \).*#\1"'${vio_pipe_file}'",#g' $vio_cfg_file
}

set_fb_pipe_file_func() {
  fb_mode=$1
  fb_res=$2
  vio_pipe_file=${vio_pipe_path}${vio_pipe_pre}${fb_res}.json
  echo "fb_mode: $fb_mode"
  echo "vio_pipe_file: $vio_pipe_file"
  sed -i 's#\("'${fb_mode}'": \).*#\1"'${vio_pipe_file}'",#g' $vio_cfg_file
}

set_multi_cam_async_pipe_file_func() {
  index=0
  for i in "$@"; do
    pipe_file=$i
    echo "pipe_file is $pipe_file"
    index=$(($index+1))
    num=$(($index*2))
    pipe_line=`cat -n ${vio_cfg_file} | grep -w "${vio_mode}" | awk '{print $1}' | sed -n ''$num'p'`
    sed -i ''${pipe_line}'s#\("'$vio_mode'": \).*#\1"'${pipe_file}'",#g' ${vio_cfg_file}
  done
}

set_multi_cam_sync_pipe_file_func() {
  index=0
  vio_mode_line=`cat -n ${vio_cfg_file} | grep -w "${vio_mode}" | awk '{print $1}' | sed -n '2p'`
  for i in "$@"; do
    pipe_file=$i
    echo "pipe_file is $pipe_file"
    index=$(($index+1))
    pipe_line=$(($vio_mode_line+$index))
    sed -i ''${pipe_line}'s#[^ ].*#"'${pipe_file}'",#g' ${vio_cfg_file}
  done
  #delete comma in multi sync pipe mode
  sed -i ''${pipe_line}'s#,##g' ${vio_cfg_file}
}

sensor_setting_func() {
  platform=$1
  sensor=$2
  vio_pipe_file=$3
  vin_vps_mode=$4
  i2c_bus=$5
  mipi_index=$6
  sensor_port=$7

  param_num=$#
  if [ $param_num -eq 3 ];then
    # set defalut sensor setting
    if [ $sensor == "imx327" ]; then
      # set vin_vps_mode and sensor port
      sed -i 's#\("vin_vps_mode": \).*#\1'0',#g' $vio_pipe_file
      sed -i 's#\("sensor_port": \).*#\1'0',#g' $vio_pipe_file
      # set i2c5 bus , mipi host1, sensor_port
      sed -i 's#\("i2c_bus": \).*#\1'5',#g' $vio_pipe_file
      sed -i 's#\("host_index": \).*#\1'1',#g' $vio_pipe_file
      echo 1 > /sys/class/vps/mipi_host1/param/stop_check_instart
    elif [ $sensor == "os8a10" -o $sensor == "os8a10_1080p" ]; then
      # set vin_vps_mode and sensor port
      sed -i 's#\("vin_vps_mode": \).*#\1'1',#g' $vio_pipe_file
      sed -i 's#\("sensor_port": \).*#\1'0',#g' $vio_pipe_file
      if [ $platform == "x3dev" ]; then
        # set i2c5 bus and mipi host1
        sed -i 's#\("i2c_bus": \).*#\1'5',#g' $vio_pipe_file
        sed -i 's#\("host_index": \).*#\1'1',#g' $vio_pipe_file
      elif [ $platform == "x3sdb" ]; then
        # set i2c2 bus and mipihost1
        sed -i 's#\("i2c_bus": \).*#\1'2',#g' $vio_pipe_file
        sed -i 's#\("host_index": \).*#\1'1',#g' $vio_pipe_file
      fi
    elif [ $sensor == "s5kgm1sp" -o $sensor == "s5kgm1sp_2160p" ]; then
      # set vin_vps_mode and sensor port
      sed -i 's#\("vin_vps_mode": \).*#\1'0',#g' $vio_pipe_file
      sed -i 's#\("sensor_port": \).*#\1'0',#g' $vio_pipe_file
      if [ $platform == "x3dev" ]; then
        # set i2c4 bus and mipi host1
        sed -i 's#\("i2c_bus": \).*#\1'4',#g' $vio_pipe_file
        sed -i 's#\("host_index": \).*#\1'1',#g' $vio_pipe_file
        echo 1 > /sys/class/vps/mipi_host1/param/stop_check_instart
      elif [ $platform == "x3sdb" ]; then
        # set i2c2 bus and mipi host1
        sed -i 's#\("i2c_bus": \).*#\1'2',#g' $vio_pipe_file
        sed -i 's#\("host_index": \).*#\1'1',#g' $vio_pipe_file
        echo 1 > /sys/class/vps/mipi_host1/param/stop_check_instart
      fi
    fi
  fi

  # set new sensor settings
  if [ -n "$i2c_bus" ];then
    echo "i2c_bus:$i2c_bus"
    sed -i 's#\("i2c_bus": \).*#\1'$i2c_bus',#g' $vio_pipe_file
  fi
  if [ -n "$mipi_index" ];then
    echo "mipi_index:$mipi_index"
    sed -i 's#\("host_index": \).*#\1'$mipi_index',#g' $vio_pipe_file
    if [ $sensor == "imx327" ];then
      echo 1 > /sys/class/vps/mipi_host${mipi_index}/param/stop_check_instart
    fi
  fi
  if [ -n "$vin_vps_mode" ];then
    echo "vin_vps_mode:$vin_vps_mode"
    sed -i 's#\("vin_vps_mode": \).*#\1'$vin_vps_mode',#g' $vio_pipe_file
  fi
  if [ -n "$sensor_port" ];then
    echo "sensor_port:$sensor_port"
    sed -i 's#\("sensor_port": \).*#\1'$sensor_port',#g' $vio_pipe_file
  fi

}

sensor_cfg_func() {
  platform=$1
  local sensor=$2
  echo "platform is $platform"
  if [ $sensor == "imx327" ]; then
    echo "sensor is imx327, default sensor output resolution 1080P, 1080P X3 JPEG Codec..."
    visual_cfg_func 0 1920 1080 -1 0 -1
    solution_cfg_func 2
  elif [ $sensor == "os8a10" -o $sensor == "os8a10_1080p" ]; then
    if [ $sensor == "os8a10" ];then
      echo "sensor is os8a10, default resolution 8M, 1080P X3 JPEG Codec..."
      visual_cfg_func 4 1920 1080 5 4 0
      solution_cfg_func 8
    elif [ $sensor == "os8a10_1080p" ];then
      echo "sensor is os8a10_1080p, default resolution 1080P, 1080P X3 JPEG Codec..."
      visual_cfg_func 0 1920 1080 -1 0 -1
      solution_cfg_func 2
    fi
    echo start > /sys/devices/virtual/graphics/iar_cdev/iar_test_attr
    echo device > /sys/devices/platform/soc/b2000000.usb/role
    echo soc:usb-id > /sys/bus/platform/drivers/extcon-usb-gpio/unbind
    service adbd stop
    /etc/init.d/usb-gadget.sh start uvc-hid
    echo 0 > /proc/sys/kernel/printk
    echo 0xc0120000 > /sys/bus/platform/drivers/ddr_monitor/axibus_ctrl/all
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/read_qos_ctrl/all
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/write_qos_ctrl/all
    if [ $platform == "x3dev" ]; then
      echo ""
    elif [ $platform == "x3sdb" ]; then
      # reset os8a10 mipi cam
      echo 111 > /sys/class/gpio/export
      echo out > /sys/class/gpio/gpio111/direction
      echo 0 > /sys/class/gpio/gpio111/value
      sleep 0.2
      echo 1 > /sys/class/gpio/gpio111/value
      # enable mclk output to os8a10 sensor in x3sdb(mipihost0)
      echo 1 > /sys/class/vps/mipi_host0/param/snrclk_en
      echo 24000000 > /sys/class/vps/mipi_host0/param/snrclk_freq
      # enable mclk output to os8a10 sensor in x3sdb(mipihost1)
      echo 1 > /sys/class/vps/mipi_host1/param/snrclk_en
      echo 24000000 > /sys/class/vps/mipi_host1/param/snrclk_freq
    fi
  elif [ $sensor == "s5kgm1sp" -o $sensor == "s5kgm1sp_2160p" ]; then
    if [ $sensor == "s5kgm1sp" ];then
      echo "sensor is s5kgm1sp, default resolution 12M, 1024*768 X3 JPEG Codec..."
      visual_cfg_func 5 1024 768 -1 -1 -1
      solution_cfg_func 12
    elif [ $sensor == "s5kgm1sp_2160p" ];then
      echo "sensor is s5kgm1sp, default resolution 8M, 1080P X3 JPEG Codec..."
      visual_cfg_func 4 1920 1080 5 4 0
      solution_cfg_func 8
    fi
    echo start > /sys/devices/virtual/graphics/iar_cdev/iar_test_attr
    echo device > /sys/devices/platform/soc/b2000000.usb/role
    echo soc:usb-id > /sys/bus/platform/drivers/extcon-usb-gpio/unbind
    service adbd stop
    /etc/init.d/usb-gadget.sh start uvc-hid
    echo 0 > /proc/sys/kernel/printk
    echo 0xc0020000 > /sys/bus/platform/drivers/ddr_monitor/axibus_ctrl/all
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/read_qos_ctrl/all
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/write_qos_ctrl/all
  elif [ $sensor == "usb_cam_1080p" ]; then
    echo "usb_cam start, default resolution 1080P..."
    echo host > /sys/devices/platform/soc/b2000000.usb/role
    service adbd stop
    solution_cfg_func 2
    visual_cfg_func 0 1920 1080 -1 0 -1
  elif [ $sensor == "ov10635_960" ]; then
    echo "ov10635_960 start, default resolution 720P..."
    echo 0 > /sys/class/vps/mipi_host0/param/stop_check_instart
    echo 0 > /sys/class/vps/mipi_host2/param/stop_check_instart
    # TODO
    visual_cfg_func 0 1280 720 -1 0 -1
  elif [ $sensor == "f37_1080p" ]; then
    echo "sensor is f37_1080p, default resolution 1080P, 1080P X3 JPEG Codec..."
    visual_cfg_func 0 1920 1080 -1 0 -1
    solution_cfg_func 2
    if [ $platform == "x3dev" ]; then
      echo ""
    elif [ $platform == "x3sdb" ]; then
      # reset f37_1080p mipi cam
      echo 119 > /sys/class/gpio/export
      echo out > /sys/class/gpio/gpio119/direction
      echo 0 > /sys/class/gpio/gpio119/value
      sleep 0.2
      echo 1 > /sys/class/gpio/gpio119/value
      # enable mclk output to f37_1080p sensor in x3sdb(mipihost0)
      echo 1 > /sys/class/vps/mipi_host0/param/snrclk_en
      echo 24000000 > /sys/class/vps/mipi_host0/param/snrclk_freq
      # enable mclk output to os8a10 sensor in x3sdb(mipihost1)
      echo 1 > /sys/class/vps/mipi_host1/param/snrclk_en
      echo 24000000 > /sys/class/vps/mipi_host1/param/snrclk_freq
    fi
    # support uvc device
    echo start > /sys/devices/virtual/graphics/iar_cdev/iar_test_attr
    echo device > /sys/devices/platform/soc/b2000000.usb/role
    echo soc:usb-id > /sys/bus/platform/drivers/extcon-usb-gpio/unbind
    service adbd stop
    /etc/init.d/usb-gadget.sh start uvc-hid
  else
    echo "error! sensor" $sensor "is not supported"
  fi
}

fb_cfg_func() {
  platform=$1
  fb_res=$2
  vio_pipe_file=$3
  echo "platform is $platform"
  echo "fb_res is $fb_res"
  echo "vio_cfg_file is $vio_cfg_file"
  echo "vio_pipe_file is $vio_pipe_file"
  echo device > /sys/devices/platform/soc/b2000000.usb/role
  echo soc:usb-id > /sys/bus/platform/drivers/extcon-usb-gpio/unbind
  service adbd stop
  /etc/init.d/usb-gadget.sh start uvc-hid
  if [ $fb_res == "1080_fb" ];then
    echo "resolution 1080P feedback start, 1080P X3 JPEG Codec..."
    solution_cfg_func 2
    visual_cfg_func 0 1920 1080 -1 0 -1
  elif [ $fb_res == "2160_fb" ];then
    echo "resolution 2160P feedback start, 1080P X3 JPEG Codec..."
    solution_cfg_func 8
    visual_cfg_func 4 1920 1080 5 4 0
  fi
}

choose_j3_single_cam_func() {
  vio_cfg_file=$1
  data_source_num=1
  echo -e 'Choose lunch single cam sensor menu...pick a combo:'
  echo -e '\t1. single camera: ov10635_960, default 720P'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$sensor_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ];then
    aNum=$sensor_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:ov10635_960 \033[0m"
      vio_pipe_pre="ov10635_960_normal/"
      sensor=ov10635_yuv_720p_960_pipe0
      vio_mode="apa_camera"
      set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_cfg_func $platform "ov10635_960"
      ;;
    *) echo -e "\033[31m You choose unsupported single cam mode \033[0m"
      exit
      ;;
  esac
}

choose_x3_single_cam_func() {
  vio_cfg_file=$1
  vio_mode="panel_camera"
  data_source_num=1
  echo -e 'Choose lunch single cam sensor menu...pick a combo:'
  echo -e '\t1. single camera: imx327, default 1080P'
  echo -e '\t2. single camera: os8a10, default 2160P'
  echo -e '\t3. single camera: os8a10_1080p, default 1080P'
  echo -e '\t4. single camera: s5kgm1sp,  default 4000*3000'
  echo -e '\t5. single camera: s5kgm1sp_2160p, default 2160P'
  echo -e '\t6. single camera: usb_cam, default 1080P'
  echo -e '\t7. single camera: f37_1080p, default 1080P'
  echo -e 'Which would you like? '
  set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
  if [ x"$run_mode" == x"ut" ];then
    aNum=$sensor_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ];then
    aNum=$sensor_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:imx327 \033[0m"
      sensor=imx327
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_setting_func $platform $sensor ${vio_pipe_file}
      sensor_cfg_func $platform $sensor
      ;;
    2)  echo -e "\033[33m You choose 2:os8a10 \033[0m"
      sensor=os8a10
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_setting_func $platform $sensor ${vio_pipe_file}
      sensor_cfg_func $platform $sensor
      ;;
    3)  echo -e "\033[33m You choose 3:os8a10_1080p \033[0m"
      sensor=os8a10_1080p
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_setting_func $platform $sensor ${vio_pipe_file}
      sensor_cfg_func $platform $sensor
      ;;
    4)  echo -e "\033[33m You choose 4:s5kgm1sp \033[0m"
      sensor=s5kgm1sp
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_setting_func $platform $sensor ${vio_pipe_file}
      sensor_cfg_func $platform $sensor
      ;;
    5)  echo -e "\033[33m You choose 5:s5kgm1sp_2160p \033[0m"
      sensor=s5kgm1sp_2160p
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_setting_func $platform $sensor ${vio_pipe_file}
      sensor_cfg_func $platform $sensor
      ;;
    6)  echo -e "\033[33m You choose 6:usb_cam \033[0m"
      vio_mode="usb_cam"
      sensor=usb_cam_1080p
      set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_cfg_func $platform $sensor
      ;;
    7)  echo -e "\033[33m You choose 7:f37_1080p \033[0m"
      sensor=f37_1080p
      set_cam_pipe_file_func $vio_mode $sensor
      sensor_setting_func $platform $sensor ${vio_pipe_file}
      sensor_cfg_func $platform $sensor
      ;;
    *) echo -e "\033[31m You choose unsupported single cam mode \033[0m"
      exit
      ;;
  esac
}

choose_fb_mode_func() {
  vio_cfg_file=$1
  fb_res=$2
  data_source_num=1

  vio_pipe_file=${vio_pipe_path}${vio_pipe_pre}${fb_res}.json
  echo -e 'Choose lunch feedback source mode menu...pick a combo:'
  echo -e '\t1. cached feedback source'
  echo -e '\t2. jpeg feedback source'
  echo -e '\t3. nv12 feedback source'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$fb_mode_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ]; then
    aNum=$fb_mode_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:cache \033[0m"
      fb_mode="cached_image_list"
      ;;
    2)  echo -e "\033[33m You choose 2:jpeg \033[0m"
      fb_mode="jpeg_image_list"
      sed -i 's#\("file_path": \).*#\1"'${fb_jpg_list}'",#g' $vio_cfg_file
      ;;
    3)  echo -e "\033[33m You choose 3:nv12 \033[0m"
      fb_mode="nv12_image_list"
      sed -i 's#\("file_path": \).*#\1"'${fb_nv12_list}'",#g' $vio_cfg_file
      ;;
    *) echo -e "\033[31m You choose unsupported fb mode \033[0m"
      exit
      ;;
  esac
  set_data_source_func ${vio_cfg_file} ${fb_mode} ${data_source_num}
  set_fb_pipe_file_func $fb_mode $fb_res
  fb_cfg_func $platform $fb_res $vio_pipe_file
}

choose_fb_cam_func() {
  vio_cfg_file=$1
  fb_res=$2
  data_source_num=2
  cam_mode="panel_camera"

  echo -e 'Choose lunch fb_cam mode menu...pick a combo:'
  echo -e '\t1. fb_cam: fb+imx327'
  echo -e '\t2. fb_cam: fb+os8a10'
  echo -e '\t3. fb_cam: fb+os8a10_1080p'
  echo -e '\t4. fb_cam: fb+s5kgm1sp'
  echo -e '\t5. fb_cam: fb+s5kg1spm_2160p'
  echo -e 'Which would you like? '
  read aNum
  case $aNum in
    1)  echo -e "\033[33m You choose 1:fb+imx327 \033[0m"
      sensor="imx327"
      ;;
    2)  echo -e "\033[33m You choose 2:fb+os8a10 \033[0m"
      sensor="os8a10"
      ;;
    3)  echo -e "\033[33m You choose 3:fb+os8a10_1080p \033[0m"
      sensor="os8a10_1080p"
      ;;
    4)  echo -e "\033[33m You choose 4:fb+s5kgm1sp \033[0m"
      sensor="s5kgm1sp"
      ;;
    5)  echo -e "\033[33m You choose 5:fb+s5kgm1sp_2160p \033[0m"
      sensor="s5kgm1sp_2160p"
      ;;
    *) echo -e "\033[31m You choose unsupported fb_cam mode \033[0m"
      exit
      ;;
  esac
  choose_fb_mode_func ${vio_cfg_file} ${fb_res}
  vio_pipe_file=${vio_pipe_path}${vio_pipe_pre}${sensor}.json
  set_fb_cam_data_source_func ${vio_cfg_file} ${fb_mode} ${cam_mode} ${data_source_num}
  set_cam_pipe_file_func $cam_mode $sensor
  sensor_setting_func $platform $sensor ${vio_pipe_file}
  sensor_cfg_func $platform $sensor
}

check_vio_pipe_file_is_exist() {
  for i in "$@"; do
    pipe_file=$i
    if [ ! -f $pipe_file ];then
      echo "vio_pipe_file:$pipe_file is not exist!"
      exit
    fi
  done
}

set_j3_multi_fb_setting_func() {
  vio_cfg_file=$1
  multi_mode=$2
  echo -e 'Choose lunch multi fb sensor menu...pick a combo:'
  echo -e '\t1. multi feedback: fb_nv12_720p_4pipes'
  echo -e 'Which would you like? '
  read aNum
  case $aNum in
    1)  echo -e "\033[33m You choose 1:fb_nv12_720p_4pipes \033[0m"
      vio_pipe_pre="multi_feedback_720p/"
      fb0="fb_720p_pipe0"
      fb1="fb_720p_pipe1"
      fb2="fb_720p_pipe2"
      fb3="fb_720p_pipe3"
      vio_pipe0_file=${vio_pipe_path}${vio_pipe_pre}${fb0}.json
      vio_pipe1_file=${vio_pipe_path}${vio_pipe_pre}${fb1}.json
      vio_pipe2_file=${vio_pipe_path}${vio_pipe_pre}${fb2}.json
      vio_pipe3_file=${vio_pipe_path}${vio_pipe_pre}${fb3}.json
      # check pipe file
      check_vio_pipe_file_is_exist ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      if [ $multi_mode == "sync" ];then
        vio_mode="multi_feedback_produce"
        data_source_num=1
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_sync_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      elif [ $multi_mode == "async" ];then
        echo -e "\033[31m You choose unsupported j3 multi async fb mode \033[0m"
        exit
      fi
      ;;
    *) echo -e "\033[31m You choose unsupported j3 multi fb mode \033[0m"
      exit
      ;;
  esac
}

set_j3_multi_cam_setting_func() {
  multi_mode=$1
  echo -e 'Choose lunch multi cam sensor menu...pick a combo:'
  echo -e '\t1. multi camera: ov10635_960_4pipes'
  echo -e '\t2. multi camera: ov10635_960_splice_4pipes'
  echo -e 'Which would you like? '
  read aNum
  case $aNum in
    1)  echo -e "\033[33m You choose 1:ov10635_960_4pipes \033[0m"
      vio_pipe_pre="ov10635_960_normal/"
      sensor0="ov10635_yuv_720p_960_pipe0"
      sensor1="ov10635_yuv_720p_960_pipe1"
      sensor2="ov10635_yuv_720p_960_pipe2"
      sensor3="ov10635_yuv_720p_960_pipe3"
      vio_pipe0_file=${vio_pipe_path}${vio_pipe_pre}${sensor0}.json
      vio_pipe1_file=${vio_pipe_path}${vio_pipe_pre}${sensor1}.json
      vio_pipe2_file=${vio_pipe_path}${vio_pipe_pre}${sensor2}.json
      vio_pipe3_file=${vio_pipe_path}${vio_pipe_pre}${sensor3}.json
      # check pipe file
      check_vio_pipe_file_is_exist ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      # set multi cam mode
      if [ $multi_mode == "sync" ];then
        vio_mode="multi_apa_camera"
        data_source_num=1
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_sync_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      elif [ $multi_mode == "async" ];then
        vio_mode="apa_camera"
        data_source_num=4
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_async_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      fi
      sensor_cfg_func $platform "ov10635_960"
      ;;
    2)  echo -e "\033[33m You choose 2:ov10635_960_splice_4pipes \033[0m"
      vio_pipe_pre="ov10635_960_splice/"
      sensor0="ov10635_yuv_720p_960_pipe0"
      sensor1="ov10635_yuv_720p_960_pipe1"
      sensor2="ov10635_yuv_720p_960_pipe2"
      sensor3="ov10635_yuv_720p_960_pipe3"
      vio_pipe0_file=${vio_pipe_path}${vio_pipe_pre}${sensor0}.json
      vio_pipe1_file=${vio_pipe_path}${vio_pipe_pre}${sensor1}.json
      vio_pipe2_file=${vio_pipe_path}${vio_pipe_pre}${sensor2}.json
      vio_pipe3_file=${vio_pipe_path}${vio_pipe_pre}${sensor3}.json
      # check pipe file
      check_vio_pipe_file_is_exist ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      # set multi cam mode
      if [ $multi_mode == "sync" ];then
        vio_mode="multi_apa_camera"
        data_source_num=1
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_sync_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      elif [ $multi_mode == "async" ];then
        vio_mode="apa_camera"
        data_source_num=4
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_async_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file} ${vio_pipe2_file} ${vio_pipe3_file}
      fi
      sensor_cfg_func $platform "ov10635_960"
      ;;
    *) echo -e "\033[31m You choose unsupported j3 multi cam mode \033[0m"
      exit
      ;;
  esac
}

set_x3_multi_cam_setting_func() {
  multi_mode=$1
  echo -e 'Choose lunch multi cam sensor menu...pick a combo:'
  echo -e '\t1. multi camera: imx327+imx327'
  echo -e '\t2. multi camera: os8a10+os8a10'
  echo -e 'Which would you like? '
  read aNum
  case $aNum in
    1)  echo -e "\033[33m You choose 1:imx327+imx327 \033[0m"
      sensor="imx327"
      sensor0="imx327_pipe0"
      sensor1="imx327_pipe1"
      vio_pipe_file=${vio_pipe_path}${vio_pipe_pre}${sensor}.json
      vio_pipe0_file=${vio_pipe_path}${vio_pipe_pre}${sensor0}.json
      vio_pipe1_file=${vio_pipe_path}${vio_pipe_pre}${sensor1}.json
      # check pipe file
      check_vio_pipe_file_is_exist ${vio_pipe_file}
      cp ${vio_pipe_file} ${vio_pipe0_file}
      cp ${vio_pipe_file} ${vio_pipe1_file}
      # set multi cam mode
      if [ $multi_mode == "sync" ];then
        vio_mode="dual_panel_camera"
        data_source_num=1
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_async_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file}
      elif [ $multi_mode == "async" ];then
        vio_mode="panel_camera"
        data_source_num=2
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_sync_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file}
      fi
      #imx327 sensor0 settings
      vin_vps_mode=2
      i2c_bus=5
      mipi_index=1
      sensor_port=0
      sensor_setting_func $platform $sensor ${vio_pipe0_file} ${vin_vps_mode} ${i2c_bus} ${mipi_index} ${sensor_port}
      #imx327 sensor1 settings
      vin_vps_mode=2
      i2c_bus=0
      mipi_index=0
      sensor_port=1
      sensor_setting_func $platform $sensor ${vio_pipe1_file} ${vin_vps_mode} ${i2c_bus} ${mipi_index} ${sensor_port}
      # imx327 sensor platform config
      sensor_cfg_func $platform $sensor
      ;;
    2)  echo -e "\033[33m You choose 2:os8a10+os8a10 \033[0m"
      sensor="os8a10"
      sensor0="os8a10_pipe0"
      sensor0="os8a10_pipe1"
      vio_pipe_file=${vio_pipe_path}${vio_pipe_pre}${sensor}.json
      vio_pipe0_file=${vio_pipe_path}${vio_pipe_pre}${sensor0}.json
      vio_pipe1_file=${vio_pipe_path}${vio_pipe_pre}${sensor1}.json
      # check pipe file
      check_vio_pipe_file_is_exist ${vio_pipe_file}
      cp ${vio_pipe_file} ${vio_pipe0_file}
      cp ${vio_pipe_file} ${vio_pipe1_file}
      # set multi cam mode
      if [ $multi_mode == "sync" ];then
        vio_mode="dual_panel_camera"
        data_source_num=1
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_async_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file}
      elif [ $multi_mode == "async" ];then
        vio_mode="panel_camera"
        data_source_num=2
        set_data_source_func ${vio_cfg_file} ${vio_mode} ${data_source_num}
        set_multi_cam_sync_pipe_file_func ${vio_pipe0_file} ${vio_pipe1_file}
      fi
      #os8a10 sensor1 settings
      vin_vps_mode=3
      i2c_bus=5
      mipi_index=1
      sensor_port=0
      sensor_setting_func $platform $sensor ${vio_pipe0_file} ${vin_vps_mode} ${i2c_bus} ${mipi_index} ${sensor_port}
      #os8a10 sensor2 settings
      vin_vps_mode=3
      i2c_bus=0
      mipi_index=0
      sensor_port=1
      sensor_setting_func $platform $sensor ${vio_pipe1_file} ${vin_vps_mode} ${i2c_bus} ${mipi_index} ${sensor_port}
      # os8a10 sensor platform config
      sensor_cfg_func $platform $sensor
      ;;
    *) echo -e "\033[31m You choose unsupported x3 multi cam mode \033[0m"
      exit
      ;;
  esac
}

choose_multi_cam_func() {
  vio_cfg_file=$1
  multi_mode=$2
  if [ $platform == "x3dev" -o $platform == "x3sdb" ];then
    set_x3_multi_cam_setting_func $multi_mode
  elif [ $platform == "j3dev" ];then
    set_j3_multi_cam_setting_func $multi_mode
  else
    echo -e "\033[31m You choose unsupported platform:$platform in multi cam mode \033[0m"
    exit
  fi
}

choose_fb_res_func() {
  echo -e 'Choose lunch fb resolution menu...pick a combo:'
  echo -e '\t1. 1080p feedback'
  echo -e '\t2. 2160p feedback'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$fb_res_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ]; then
    aNum=$fb_res_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:1080p_feedback \033[0m"
      fb_res=1080_fb
      nv12_list_name="name_nv12_1080p.list"
      jpg_list_name="name_jpg_1080p.list"
      sed -i 's#\("pad_width": \).*#\1'1920',#g' $vio_cfg_file
      sed -i 's#\("pad_height": \).*#\1'1080',#g' $vio_cfg_file
      ;;
    2)  echo -e "\033[33m You choose 2:2160p_feedback \033[0m"
      fb_res=2160_fb
      nv12_list_name="name_nv12_2160p.list"
      jpg_list_name="name_jpg_2160p.list"
      sed -i 's#\("pad_width": \).*#\1'3840',#g' $vio_cfg_file
      sed -i 's#\("pad_height": \).*#\1'2160',#g' $vio_cfg_file
      ;;
    *) echo -e "\033[31m You choose unsupported feedback resolution \033[0m"
      exit
      ;;
  esac
  cp ${fb_pic_path}${nv12_list_name} ${fb_nv12_list}
  cp ${fb_pic_path}${jpg_list_name} ${fb_jpg_list}
}

choose_j3_viotype_func() {
  echo -e 'Choose lunch j3 vio type menu...pick a combo:'
  echo -e '\t1. single cam'
  echo -e '\t2. single feedback'
  echo -e '\t3. multi cam sync'
  echo -e '\t4. multi cam async'
  echo -e '\t5. multi feedback sync'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$vio_type_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ]; then
    aNum=$vio_type_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:single_cam \033[0m"
      vio_cfg_file=${vio_cfg_name}.j3dev.cam
      choose_j3_single_cam_func ${vio_cfg_file}
      ;;
    2)  echo -e "\033[33m You choose 2:single_fb \033[0m"
      vio_cfg_file=${vio_cfg_name}.j3dev.fb
      choose_fb_res_func
      choose_fb_mode_func ${vio_cfg_file} ${fb_res}
      ;;
    3)  echo -e "\033[33m You choose 3:multi_cam_sync \033[0m"
      vio_cfg_file=${vio_cfg_name}.j3dev.multi_cam_sync
      multi_mode="sync"
      choose_multi_cam_func ${vio_cfg_file} ${multi_mode}
      ;;
    4)  echo -e "\033[33m You choose 4:multi_cam_async \033[0m"
      vio_cfg_file=${vio_cfg_name}.j3dev.multi_cam_async
      multi_mode="async"
      choose_multi_cam_func ${vio_cfg_file} ${multi_mode}
      ;;
    5)  echo -e "\033[33m You choose 5:multi_fb_sync \033[0m"
      vio_cfg_file=${vio_cfg_name}.j3dev.multi_fb_sync
      multi_mode="sync"
      set_j3_multi_fb_setting_func ${vio_cfg_file} ${multi_mode}
      ;;
    *) echo -e "\033[31m You choose unsupported j3 vio type mode \033[0m"
      exit
      ;;
  esac
}

choose_x3_viotype_func() {
  echo -e 'Choose lunch x3 vio type menu...pick a combo:'
  echo -e '\t1. single cam'
  echo -e '\t2. single feedback'
  echo -e '\t3. multi cam sync'
  echo -e '\t4. multi cam async'
  echo -e '\t5. single cam + single feedback'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$vio_type_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ]; then
    aNum=$vio_type_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:single_cam \033[0m"
      vio_cfg_file=${vio_cfg_name}.x3dev.cam
      choose_x3_single_cam_func ${vio_cfg_file}
      ;;
    2)  echo -e "\033[33m You choose 2:single_fb \033[0m"
      vio_cfg_file=${vio_cfg_name}.x3dev.fb
      choose_fb_res_func
      choose_fb_mode_func ${vio_cfg_file} ${fb_res}
      ;;
    3)  echo -e "\033[33m You choose 3:multi_cam_sync \033[0m"
      vio_cfg_file=${vio_cfg_name}.x3dev.multi_cam_sync
      multi_mode="sync"
      choose_multi_cam_func ${vio_cfg_file} ${multi_mode}
      ;;
    4)  echo -e "\033[33m You choose 4:multi_cam_async \033[0m"
      vio_cfg_file=${vio_cfg_name}.x3dev.multi_cam_async
      multi_mode="async"
      choose_multi_cam_func ${vio_cfg_file} ${multi_mode}
      ;;
    5)  echo -e "\033[33m You choose 5:fb_cam \033[0m"
      vio_cfg_file=${vio_cfg_name}.x3dev.fb_cam
      choose_fb_res_func
      choose_fb_cam_func ${vio_cfg_file} ${fb_res}
      ;;
    *) echo -e "\033[31m You choose unsupported x3 vio type mode \033[0m"
      exit
      ;;
  esac
}

choose_platform_func() {
  pass_viotype_ck=$1
  echo "pass_viotype_ck is ${pass_viotype_ck}"
  echo -e 'Choose lunch platform menu...pick a combo:'
  echo -e '\t1. x2_96board'
  echo -e '\t2. x2_2610'
  echo -e '\t3. x3_sdb'
  echo -e '\t4. x3_dev'
  echo -e '\t5. j3_dev'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$platform_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ]; then
    aNum=$platform_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:x2 96board \033[0m"
      platform="96board"
      visual_cfg_func 4 960 540 -1 -1 -1
      vio_cfg_file=${vio_cfg_name}.96board
      ;;
    2)  echo -e "\033[33m You choose 2:x2 2610 \033[0m"
      platform="2610"
      visual_cfg_func 4 960 540 -1 -1 -1
      vio_cfg_file=${vio_cfg_name}.2610
      ;;
    3)  echo -e "\033[33m You choose 3:x3sdb \033[0m"
      platform="x3sdb"
      echo  1000000000 > /sys/class/devfreq/devfreq1/userspace/set_freq
      echo 105000 >/sys/devices/virtual/thermal/thermal_zone0/trip_point_1_temp
      echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
      # use dynamic ion alloc memory
      echo 0 > /sys/class/misc/ion/cma_carveout_size
      if [ -z "${pass_viotype_ck}" ]; then
        choose_x3_viotype_func
      fi
      ;;
    4)  echo -e "\033[33m You choose 4:x3dev \033[0m"
      platform="x3dev"
      echo  1000000000 > /sys/class/devfreq/devfreq1/userspace/set_freq
      echo 105000 >/sys/devices/virtual/thermal/thermal_zone0/trip_point_1_temp
      echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
      # use dynamic ion alloc memory
      echo 0 > /sys/class/misc/ion/cma_carveout_size
      if [ -z "${pass_viotype_ck}" ]; then
        choose_x3_viotype_func
      fi
      ;;
    5)  echo -e "\033[33m You choose 5:j3dev \033[0m"
      platform="j3dev"
      vio_pipe_path="configs/vio/j3dev/"
      vio_pipe_pre=""
      if [ -z "${pass_viotype_ck}" ]; then
        choose_j3_viotype_func
      fi
      ;;
    *) echo -e "\033[31m You choose unsupported platform mode \033[0m"
      exit
      ;;
  esac
}

choose_viotest_loop_mode_func() {
  viotest_mode_aNum=1
  echo -e 'Choose lunch vio test loop mode menu...pick a combo:'
  echo -e '\t1. vio_start_stop_once'
  echo -e '\t2. vio_start_stop_loop'
  if [ x"$run_mode" == x"ut" ];then
    aNum=$viotest_mode_aNum
  elif [ x"${run_mode}" == x"cmd_normal" ]; then
    aNum=$viotest_mode_aNum
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:vio test start_stop_once \033[0m"
      vio_test_loop=0
      ;;
    2)  echo -e "\033[33m You choose 1:vio test start_stop_loop \033[0m"
      vio_test_loop=1
      ;;
    *) echo -e "\033[31m You choose unsupported vio test loop mode \033[0m"
      exit
      ;;
  esac
}

choose_solution_func() {
  log_level=$1
  if [ -z "$log_level" ];then
    log_level="i"
  fi
  echo "log_level: $log_level"
  echo "run_mode: $run_mode"
  echo -e 'Choose lunch solution menu...pick a combo:'
  echo -e '\t1.  face'
  echo -e '\t2.  face_recog'
  echo -e '\t3.  body'
  echo -e '\t4.  xbox'
  echo -e '\t5.  behavior'
  echo -e '\t7.  video_box'
  echo -e '\t8.  tv_uvc'
  echo -e '\t9.  face_body_multisource'
  echo -e '\t10. apa'
  echo -e '\t11. multi_input_hapi'
  echo -e '\t12. apa_test'
  echo -e '\t13. ssd_test'
  echo -e '\t14. vio_test'
  echo -e '\t15. multi_input_hapi_ipm'
  echo -e '\t16. matting'
  echo -e '\t17. yolov3_mobilenetv3_example'
  # echo -e '\t30. vehicle'
  # echo -e '\t31. vehicle_v2'
  # echo -e '\t32. gtest_vision_type'
  # echo -e '\t33. gtest_image_tools'
  # echo -e '\t34. gtest_grading'
  # echo -e '\t35. x2_4k'
  echo -e 'Which would you like? '
  if [ x"$run_mode" == x"ut" ];then
    aNum=$solution_mode_aNum
    echo "solution_mode_aNum is $solution_mode_aNum"
  elif [ x"${run_mode}" == x"cmd_normal" ];then
    aNum=$solution_mode_aNum
    echo "solution_mode_aNum is $solution_mode_aNum"
  else
    read aNum
  fi
  case $aNum in
    1)  echo -e "\033[33m You choose 1:face \033[0m"
      choose_platform_func
      if [ x"$run_mode" == x"ut" ];then
        ./face_solution/face_solution $vio_cfg_file ./face_solution/configs/face_solution.json ./configs/visualplugin_face.json -${log_level}
      else
        ./face_solution/face_solution $vio_cfg_file ./face_solution/configs/face_solution.json ./configs/visualplugin_face.json -${log_level} normal
      fi
      ;;
    2)  echo -e "\033[33m You choose 2:face_recog \033[0m"
      choose_platform_func
      if [ x"$run_mode" == x"ut" ];then
        ./face_solution/face_solution $vio_cfg_file ./face_solution/configs/face_recog_solution.json ./configs/visualplugin_face.json -${log_level}
      else
        ./face_solution/face_solution $vio_cfg_file ./face_solution/configs/face_recog_solution.json ./configs/visualplugin_face.json -${log_level} normal
      fi
      ;;
    3)  echo -e "\033[33m You choose 3:body \033[0m"
      choose_platform_func
      echo "vio_cfg_file: $vio_cfg_file"
      if [ x"$run_mode" == x"ut" ];then
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/body_solution.json ./configs/visualplugin_body.json -${log_level}
      else
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/body_solution.json ./configs/visualplugin_body.json -${log_level} normal
      fi
      ;;
    4)  echo -e "\033[33m You choose 4:xbox \033[0m"
      choose_platform_func
      if [ x"$run_mode" == x"ut" ];then
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/xbox_solution.json ./configs/visualplugin_body.json -${log_level}
      else
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/xbox_solution.json ./configs/visualplugin_body.json -${log_level} normal
      fi
      ;;
    5)  echo -e "\033[33m You choose 5:behavior \033[0m"
      choose_platform_func
      if [ x"$run_mode" == x"ut" ];then
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/behavior_solution.json ./configs/visualplugin_body.json -${log_level}
      else
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/behavior_solution.json ./configs/visualplugin_body.json -${log_level} normal
      fi
      ;;
    6)  echo -e "\033[33m You choose 6:gesture \033[0m"
      choose_platform_func
      if [ x"$run_mode" == x"ut" ];then
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/gesture_solution.json ./configs/visualplugin_body.json -${log_level}
      else
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/gesture_solution.json ./configs/visualplugin_body.json -${log_level} normal
      fi
      ;;
    7)  echo -e "\033[33m You choose 7:video_box \033[0m"
      if [ x"$run_mode" == x"ut" ];then
        ./video_box/video_box ./video_box/configs/video_box_config.json ./video_box/configs/visualplugin_video_box.json ./video_box/configs/ware_config.json -${log_level}
      elif [ x"$run_mode" == x"ruku" ];then
        ./video_box/video_box ./video_box/configs/video_box_config.json ./video_box/configs/visualplugin_video_box.json ./video_box/configs/ware_config.json  -${log_level} ruku ./configs/vio_config.json.x3dev.fb_face_pic
      elif [ x"$run_mode" == x"db_list" ];then
        ./video_box/video_box ./video_box/configs/video_box_config.json ./video_box/configs/visualplugin_video_box.json ./video_box/configs/ware_config.json -${log_level} db_list 
      elif [ x"$run_mode" == x"db_drop_table" ];then
        ./video_box/video_box ./video_box/configs/video_box_config.json ./video_box/configs/visualplugin_video_box.json ./video_box/configs/ware_config.json -${log_level} db_drop_table 
      else
        ./video_box/video_box ./video_box/configs/video_box_config.json ./video_box/configs/visualplugin_video_box.json ./video_box/configs/ware_config.json -${log_level} normal
      fi
      ;;
    8)  echo -e "\033[33m You choose 8:tv_uvc \033[0m"
      choose_platform_func
      echo "vio_cnfig_file: $vio_cfg_file"
      if [ x"$run_mode" == x"ut" ];then
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/dance_solution_multitask_960x544.json ./configs/visualplugin_body.json -${log_level}
      else
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/dance_solution_multitask_960x544.json ./configs/visualplugin_body.json -${log_level} normal
      fi
      ;;
    9)  echo -e "\033[33m You choose 9:face_body_multisouce \033[0m"
      choose_platform_func "pass_viotype_ck"
      vio_cfg_file=${vio_cfg_name}.$platform.fb
      if [ $platform == "x3sdb" ];then
        vio_cfg_file=${vio_cfg_name}.x3dev.fb
      fi
      echo "vio_cfg_file is ${vio_cfg_file}"
      if [ $platform == "x3dev" -o $platform == "x3sdb" ];then
        sed -i 's#\("data_source": \).*#\1"cached_image_list",#g' ${vio_cfg_file}
      fi
      if [ x"$run_mode" == x"ut" ];then
        ./face_body_multisource/face_body_multisource ${vio_cfg_file} ./face_body_multisource/configs/face_body_solution.json -${log_level}
      else
        ./face_body_multisource/face_body_multisource ${vio_cfg_file} ./face_body_multisource/configs/face_body_solution.json -${log_level} normal
      fi
      ;;
    10)  echo -e "\033[33m You choose 10:apa_vapi \033[0m"
      vio_cfg_file=./configs/hb_vio_config.json.j3dev
      echo "vio_cfg_file is ${vio_cfg_file}"
      if [ x"$run_mode" == x"ut" ];then
        ./apa/apa $vio_cfg_file ./apa/configs/apa_config.json ./apa/configs/websocket_config.json ./apa/configs/gdcplugin_config.json ./apa/configs/displayplugin_config.json -${log_level}
      else
        ./apa/apa $vio_cfg_file ./apa/configs/apa_config.json ./apa/configs/websocket_config.json ./apa/configs/gdcplugin_config.json ./apa/configs/displayplugin_config.json -${log_level} normal
      fi
      ;;
    11)  echo -e "\033[33m You choose 11:multi_input_hapi \033[0m"
      choose_platform_func
      visual_cfg_func 0 1280 720 -1 -1 -1
      if [ x"$run_mode" == x"ut" ];then
        ./multisourceinput/multisourceinput $vio_cfg_file ./apa/configs/apa_config.json ./configs/visualplugin_multisource.json -${log_level}
      else
        ./multisourceinput/multisourceinput $vio_cfg_file ./apa/configs/apa_config.json ./configs/visualplugin_multisource.json -${log_level} normal
      fi
      ;;
    12)  echo -e "\033[33m You choose 12:apa_test \033[0m"
      ./apa/multivioplugin_test
      ;;
    13)  echo -e "\033[33m You choose 13:ssd_test \033[0m"
      export LD_LIBRARY_PATH=./lib/:../lib
      cp configs/vio/vio_onsemi0230_fb.json  ./ssd_test/config/vio_config/vio_onsemi0230_fb.json
      cd ./ssd_test
      ./ssd_method_test
      ;;
    14)  echo -e "\033[33m You choose 14:vioplugin_test \033[0m"
      choose_platform_func
      choose_viotest_loop_mode_func
      ./vioplugin_test/vioplugin_sample ${vio_cfg_file} ${vio_test_loop} -${log_level}
      ;;
    15)  echo -e "\033[33m You choose 15:multi_input_hapi_ipm \033[0m"
      vio_cfg_file=./configs/vio_config.json.j3dev.multi_fb_sync.256x512
      visual_cfg_func 0 256 512 -1 -1 -1
      if [ x"$run_mode" == x"ut" ];then
        ./multisourceinput/multisourceinput $vio_cfg_file ./apa/configs/apa_ipm_config.json ./configs/visualplugin_multisource.json -${log_level}
      else
        ./multisourceinput/multisourceinput $vio_cfg_file ./apa/configs/apa_ipm_config.json ./configs/visualplugin_multisource.json -${log_level} normal
      fi
      ;;
    16)  echo -e "\033[33m You choose 16:matting \033[0m"
      choose_platform_func
      echo "vio_cnfig_file: $vio_cfg_file"
      if [ x"$run_mode" == x"ut" ];then
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/matting_solution_multitask_960x544.json ./configs/visualplugin_body.json -${log_level}
      else
        ./body_solution/body_solution $vio_cfg_file ./body_solution/configs/matting_solution_multitask_960x544.json ./configs/visualplugin_body.json -${log_level} normal
      fi
      ;;
    17)  echo -e "\033[33m You choose 17:yolvo3_mobilenetv2  \033[0m"
      choose_platform_func
      echo "vio_cnfig_file: $vio_cfg_file"
      ./yolov3_solution/yolov3_solution $vio_cfg_file ./yolov3_solution/configs/solution_yolov3.json ./configs/visualplugin_body.json -${log_level} normal
      ;;
    # 30)  echo -e "\033[33m You choose 30:vehicle \033[0m"
    #   if [ x"$run_mode" == x"ut" ];then
    #     ./vehicle_solution/vehicle_solution $vio_cfg_file ./vehicle_solution/configs/vehicle_solution.json ./configs/visualplugin_vehicle.json -${log_level}
    #   else
    #     ./vehicle_solution/vehicle_solution $vio_cfg_file ./vehicle_solution/configs/vehicle_solution.json ./configs/visualplugin_vehicle.json -${log_level} normal
    #   fi
    #   ;;
    # 31)  echo -e "\033[33m You choose 31:vehicle_v2 \033[0m"
    #   if [ x"$run_mode" == x"ut" ];then
    #     ./vehicle_solution/vehicle_solution $vio_cfg_file ./vehicle_solution/configs/vehicle_solution_v2x.json ./configs/visualplugin_vehicle.json -${log_level}
    #   else
    #     ./vehicle_solution/vehicle_solution $vio_cfg_file ./vehicle_solution/configs/vehicle_solution_v2x.json ./configs/visualplugin_vehicle.json -${log_level} normal
    #   fi
    #   ;;
    # 32)  echo -e "\033[33m You choose 32:gtest_vision_type \033[0m"
    #   ./vision_type/gtest_vision_type
    #   ;;
    # 33)  echo -e "\033[33m You choose 33:gtest_image_tools \033[0m"
    #   ./image_tools/gtest_imagetools
    #   ;;
    # 34)  echo -e "\033[33m You choose 34:gtest_grading \033[0m"
    #   cd methods/grading/
    #   ./gtest_grading
    #   ;;
    # 35)  echo -e "\033[33m You choose 35:x2_4k \033[0m"
    #   choose_platform_func
    #   if [ $platform == "96board" ]
    #   then
    #     sed -i 's#\("x2_4k": \).*#\1"configs/vio/vio_mipi2160p.json.96board",#g' configs/vio_config.json.hg.4k
    #   elif [ $platform == "2610" ]
    #   then
    #     sed -i 's#\("x2_4k": \).*#\1"configs/vio/vio_mipi2160p.json.2610",#g' configs/vio_config.json.hg.4k
    #   else
    #     echo "not support, todo"
    #   fi
    #   if [ x"$run_mode" == x"ut" ];then
    #     ./face_body_multisource/face_body_multisource configs/vio_config.json.hg.4k ./face_body_multisource/configs/face_body_solution.json
    #   else
    #     ./face_body_multisource/face_body_multisource configs/vio_config.json.hg.4k ./face_body_multisource/configs/face_body_solution.json -w normal
    #   fi
    #   ;;
    *) echo -e "\033[31m You choose unsupported solution mode \033[0m"
      exit
      ;;
  esac
}

choose_solution_func $1
# stop_nginx
