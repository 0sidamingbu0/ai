# 智能计算盒参考方案
## 介绍
智能计算盒参考方案，主要功能是接收多路IPC的RTSP码流，解码后进行智能分析，并将分析结果渲染VO输出，或者再通过RTSP推流发送出去。
## 能力

当前DEMO智能分析使用的模型是Body Solution模型，检测结果有人脸框，人脸关键点，人体框，人体关键点等信息。最大支持接收8路RTSP码流，以及2路（1080P，720P）RTSP推流。

## 编译
 ```
bash build.sh
 ```
## 打包部署包
 ```
bash deploy.sh
 ```
该脚本会在当前目录下创建deploy文件夹，里面包含通用库、VIO配置文件、模型及body_solution目录

## 运行
将部署包拷贝到板子上，即可运行。
 ```
运行run.sh，选择video_box选项即可。
 ```
## 入库
1.更换./configs/vio_face_pic的jpg图片。
2.修改name_jpg.list的文件列表。
3.运行run.sh e ruku，选择video_box选项即可。(其中e为错误级别的log，可选f/e/i/w/d)

## 删除底库
运行run.sh e db_drop_table，选择video_box选项即可。

## 查询底库
运行run.sh i db_list，选择video_box选项即可。

## 配置文件说明

1.video_box_config.json

```json
{
  "xstream_workflow_file": "./video_box/configs/body_detection.json",	// xstream配置文件
  "xstream_workflow_file_pic": "./video_box/configs/recog_flow.json",   // xstream配置文件
  "xstream_workflow_file_feature": "./video_box/configs/feature_flow.json",  // xstream配置文件
  "enable_recog": false,                      // 是否使能人脸识别功能
  "enable_profile": 0,
  "profile_log_path": "",
  "enable_result_to_json": false,
  "box_face_thr": 0.95,													// 人脸框阈值
  "box_head_thr": 0.95,													// 人头框阈值
  "box_body_thr": 0.95,													// 人体框阈值
  "lmk_thr": 0.0,
  "kps_thr": 0.50,
  "box_veh_thr": 0.995,
  "plot_fps": false,
  "rtsp_config_file": "./video_box/configs/rtsp.json",					// rtsp配置文件
  "display_config_file": "./video_box/configs/display.json",				// 显示配置文件
  "drop_frame_config_file": "./video_box/configs/drop_frames.json",  // 配置是否丢帧以及丢帧策略
  "run_smart": true                            // 是否运行智能分析功能，当不运行智能分析功能时，从网络rtsp过来的码流解码之后直接VO显示输出
}
```

2.rtsp.json

~~~json
{
  "channel_num": 4,                            					// 连接路数
  "channel0": {
    "rtsp_link": "rtsp://admin:admin123@10.96.35.66:554/0",   	// rtsp URL地址
    "tcp": 1,													// 是否使用TCP连接
    "frame_max_size": 200, 										// 码流最大包大小，单位为KB
    "save_stream": 0											// 是否保存当前码流到本地
  },
  "channel1": {
    "rtsp_link": "rtsp://admin:admin123@10.96.35.199:554/0",
    "tcp": 0,
    "frame_max_size": 200,
    "save_stream": 0
  },
  "channel2": {
    "rtsp_link": "rtsp://admin:admin123@10.96.35.66:554/0",
    "tcp": 1,
    "frame_max_size": 200,
    "save_stream": 0
  },
  "channel3": {
    "rtsp_link": "rtsp://admin:admin123@10.96.35.66:554/0",
    "tcp": 0,
    "frame_max_size": 200,
    "save_stream": 0
  }
}
~~~

3.display.json

```json
{
  "vo": {
    "enable": true,											// 是否开启VO输出
    "display_mode": 0, 									// 显示模式，0: auto, 1: 9 pictures
    "transition_support": 0,            // 是否支持动态融合
    "draw_real_time_video": false,       // 渲染视频是否实时，不丢帧
    "draw_smart": true                 // 是否渲染智能帧信息
  },
  "rtsp": {
    "stream_1080p": false,									// 是否开启1080P RTSP码流推送
    "stream_720p": false,									// 是否开启720P RTSP码流推送
    "encode_smart_info": false             // 编码码流是否包括智能信息
  }
}
```

4.visualplugin_video_box.json

```json
{
  "auth_mode": 0,								// RTSP推流是否开启密码验证功能
  "display_mode": 1,							// 设置为0开启本地转发或推流功能
  "user": "admin",								// RTSP推流用户名
  "password": "123456",							// RTSP推流密码
  "data_buf_size": 3110400,						// live555缓存区大小
  "packet_size": 102400,						// live555分包大小
  "input_h264_filename": "",					// 需要转发的码流文件名，不需要转发则置空
  "local_forward": 1							// 必须设置为1
}
```

5.drop_frames.json

```json
{
    "frame_drop": {
        "drop_frame": false,      // 是否丢帧
        "interval_frames_num": 2  // 间隔多少帧丢弃一帧
    }
}
```

## 常见问题

1.拉取码流出错

1) rtsp.json文件里面的url地址问题，注意检查url的用户名密码以及ip。

2) 设备支持的连接数目达到最大限制，不再支持继续接入。
   比如，某些设备rtsp的tcp方式最大支持一路，当前已使用tcp接入了一路，此时可修改rtsp.json对应通道的连接方式。


2.解码出错

1）目前接收码流buffer默认为200KB，如果码流帧大小超过200KB，会导致接收到的帧不完整，送给解码器后解码失败。
   码流帧大小的限制可以在rtsp.json文件里面修改。

2）送给解码器要是一帧完整的帧，包含start code（0x00 0x00 0x00 0x01）、SPS、PPS数据头。


3.程序启动失败

运行多路时，比如8路，程序启动失败，可能原因是默认ion Buffer设置太小，需要手动修改下，修改方法为 uboot 命令行下修改环境变量的，命令如下。

以配置 1GB ion 空间为例： 

~~~shell
setenv ion_size '1024' 
saveenv 
~~~

4.HDMI没有显示

开发板硬件上支持 HDMI 和 LCD 两种 display 方式，默认显示方式为LCD，若要HDMI显示需要手动在`uboot`下修改环境变量切换，切换命令如下。 
HDMI 方式： 

~~~shell
setenv bootargs earlycon console=ttyS0,921600 clk_ignore_unused video=hobot:x3sdb-hdmi kgdboc=ttyS0 
saveenv 
~~~


LCD 方式：

~~~shell
setenv bootargs earlycon console=ttyS0,921600 clk_ignore_unused video=hobot:x3sdb-mipi720p kgdboc=ttyS0 
saveenv 
~~~

5、显示模式
目前支持单画面，4画面，9画面显示的方式。默认是根据接入的路数动态切换显示画面的个数。若需要修改显示方式，可以在配置文件修改display_mode字段。

