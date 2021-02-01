# usb cam解决方案

## 介绍
1、基于manifest编译，相关依赖：
	http://gitlab.hobot.cc/iot/devices/x2solution/manifest.git
	solution_zoo.xml

2、解决方案支持两种编译和部署模式
	（1）demo模式
		只编译usb_cam_x3/sample。
		支持X3侧的自启动运行和通过AP启动两种功能。
		部署包中只有可执行程序、依赖库、配置文件和模型。
	（2）sdk模式
		编译usb_cam_x3/src下的wrapperplugin和horizon、usb_cam_x3/sdk_sample。
		部署包中包括已编译好的可执行程序、依赖库、配置文件、模型、sdk的头文件、sample程序和编译脚本。
		支持X3侧的自启动运行。

## 编译
编译脚本自动化的执行编译和打包，生成release部署包，将部署包拷贝到开发板上直接运行。
（1）demo模式
	sh BuildAll/make_usbcam_sdk.sh x3 release或sh BuildAll/make_usbcam_sdk.sh x3 release demo
（2）sdk模式
	sh BuildAll/make_usbcam_sdk.sh x3 release sdk

## 运行
（1）demo模式
	release包中的执行脚本为start.sh，直接运行脚本，根据提示选择运行的功能:
	脚本说明
	    a. 脚本命令
            sh start.sh [ os8a10 | os8a10_1080p | sc8238 | hg | usb_cam] [ cache | jpg | nv12 | buffer ] [ 1080_fb | 2160_fb ] [ face | body ]
	    b. 输入源 [ os8a10 | os8a10_1080p | sc8238 | hg | usb_cam]
	    	os8a10：os8a10 sensor，使用2160P分辨率输出
	    	os8a10_1080p：os8a10 sensor，使用1080P分辨率输出
	    	sc8238：sc8238 sensor，只支持2160P分辨率输出
	    	hg：回灌
	    	usb_cam：外接usb cam摄像头
	    c. 回灌模式 [ cache | jpg | nv12 | buffer ]
	        jpg：jpg格式图片namelist输入，vio plugin中自动加载namelist中图片，并通过软解码将jpeg格式图片转换成nv12格式，支持任意分辨率图片
	        nv12：nv12格式图片namelist输入，vio plugin中自动加载namelist中图片，支持任意分辨率图片
	        buffer：图片buffer输入，mc plugin依次读取configs/mc_config.json中"feedback_file_path"配置的namelist中图片，构建并发布图片buffer的xproto msg，
	                vio plugin订阅此msg，使用buffer生成pym用于智能分析。如果是jpeg格式图片，vio plugin通过硬件解码将jpeg格式图片转换成nv12格式
	                对于jpeg图片，当前只支持1080P和2160P格式输入
	    d. 回灌分辨率 [ 1080_fb | 2160_fb ]
	        1080_fb：将回灌的图片padding到1080p
	        2160_fb：将回灌的图片padding到2160p
	        如果不选择，默认使用1080P
	    e. workflow [ face | body ]
	        face：人脸抓拍和识别workflow
	        body：人体和手势识别workflow
	        如果不选择，默认使用body
	使用举例
	    a. start.sh os8a10
	    	os8a10 sensor作为输入 & 人体和手势识别workflow
	    b. start.sh sc8238 face
	    	sc8238 sensor作为输入 & 人脸抓拍和识别workflow
	    c. sh start.sh hg jpg 1080_fb
	        jpg格式1080p的本地图片回灌
	    d. sh start.sh hg buffer 2160_fb
	        2160p图片buffer输入
（2）sdk模式
	release包中包含可执行程序，执行脚本为start_sdk.sh，直接运行脚本，根据提示选择运行的功能。

	release包中还包含sdk的参考sample源程序、头文件、库、sample源程序和编译脚本，支持客户使用sdk做方案开发。
	sample程序的路径为release/sdk_sample，使用sample编译和运行方法：
	cd release/sdk_sample	#进入sample路径
	sh build.sh				#编译和部署
	sh start_sdk.sh			#运行

## 配置文件说明
```
1. configs/mc_config.json
{
  "enable_auto_start": 0,                       # 是否使能自启动，不使能需要通过AP拉起X3。standalone模式需要使能。
  "enable_vot": 1,                              # 是否使能VOT（HDMI）输出，使能会将智能检测结果的渲染到视频帧并通过HDMI输出展示。
  "vot_config": "./configs/vot_config.json",    # VOT模块的配置文件路径
  "enable_dump_smart": 0,                       # 是否使能dump智能检测结果。使能会按帧输出每帧智能检测结果，格式：[frame id] [框坐标、score、框类型、id] .......
                                                # [hand lmk] [模型输出的分类结果、投票结果、经过策略结果、score] [模型输出的原始15类结果]
                                                # 如果有hand，才输出21点hand lmk。如果模型输出的手势识别分类结果不是-1（-1说明模型没有输出），才输出模型输出的原始15类结果
  "enable_append_smart": 0,                     # 是否使能渲染附加智能检测结果。使能会附加渲染手势识别原始结果等信息。
  "enable_dump_img": 0,                         # 是否使能dump原图（金字塔第0层）。使能会存储原图，并且影响智能帧率。
  "save_dump_img_path": "dump_pym0_img/",       # dump原图的保存路径
  "enable_feedback": 0,                         # 是否使能回灌。如果使能，mc plugin依次读取namelist中的图片，构建并发布图片buffer的xproto msg，vio plugin中通过硬件解码将jpeg格式图片转换成nv12格式
  "feedback_file_path": "configs/vio_hg/name_jpg.list", # 回灌namelist文件名
  "desc": "enable_dump_smart: dump smart data for badcase analysis; enable_append_smart: append more oupt for vot display, e.g. gesture raw oupt, ..."
}

2. configs/vot_config.json
{
  "en_encoder": false,                      # 是否使能编码，使能会将视频帧做编码(h264编码)后输出存储
  "encoder_input": 0,                       # 解码器的输入源，0：输入智能结果渲染后的视频帧；1：输入原始视频帧（用于数据采集）
  "encoder_output_save_file": "draw.h264",  # 视频编码存储文件名
  "en_bgr_convert": false,                  # 是否使能渲染时图片格式转换。使能会将图片转成bgr后彩色渲染，并且会影响智能帧率
  "en_fps_draw": true,                      # 是否渲染智能fps
  "en_gesture_val_draw": false,             # 是否渲染手势识别结果数值，包括模型原始输出，投票输出和策略输出
  "en_handid_draw": false,                  # 是否渲染人手ID
  "desc": "en_encoder: input img to encoder and save output(h264); encoder_input: 0: input img with smart drawing to encoder, 1: input raw img to encoder; encoder_output_save_file: saved file name; en_bgr_convert: convert nv12 to bgr and plot on bgr img; en_fps_draw: draw fps on img"
}

## 编码视频使用说明
当前只支持h264格式视频编码输出，可以使用暴风影音、vlc和potplayer等视频播放软件直接播放。
可以使用ffmpeg工具将h264格式视频做格式转换：
1. 转换成mp4格式
    ffmpeg -i $IN_VIDEO_NAME.h264 -vcodec copy -f mp4 $OUT_VIDEO_NAME.mp4
2. 转换成图片
    ffmpeg -i $VIDEO_NAME -s 1920x1080 -q:v 2 $TARGET_FOLDER/video-%8d.jpeg
    可以指定输出图片的分辨率,举例输出分辨率为1080P
对于渲染后的编码视频，转换成单张图片可以用于算法badcase分析。对于渲染前的原图编码视频，转换成图片可以用于回灌。




# 智能取景以及C位发言人功能

## 介绍
智能取景：主要功能是智能识别参与会议人员的位置，智能的显示所有与会人员，争取将所有与会人员显示在一个画面。
发言人C位:  智能追踪会议发言人的位置，显示发言人位置，将发言人区域放大。

## 配置文件说明
1.roi_config.json

```json
{
    "enable_auto_start": 1,             // 是否自动启动
    "enable_vot": 1,                    // 是否启动vot模块渲染
	"enable_intelligent_tracking": 0,   // 是否启动发言人智能追踪
    "view_roi_update_thr": 0.85,        // 智能取景框更新阈值，取值范围[0-1]，越接近1，框更新越频繁
    "view_expansion_coefficient": 1.2,  // 智能取景框外扩系数，取值范围[1-2]，输出框放大倍数
    "view_transition_coefficient": 16,  // 智能取景框过度速度系数，取值范围[4-64]，值越大，框过度越慢
    "track_roi_update_thr": 0.75,       // 智能追踪框更新阈值，取值范围[0-1]，越接近1，框更新越频繁
    "track_expansion_coefficient": 1.2, // 智能追踪框外扩系数，取值范围[1-2]，输出框放大倍数
    "track_roi_display": 2,             // 智能追踪框显示个数，取值范围[2-3]，实现要显示的智能追踪框数
    "keep_ratio": true                  // 是否保证框比例和原图一致
}
```
注意：此处若使能vot, 则mc_config.json里面的vot需要禁用
