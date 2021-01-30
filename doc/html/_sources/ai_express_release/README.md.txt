XStream
=======

# 介绍

XStream 发版包编译、运行介绍。

# 编译
## 编译环境

需提前准备好交叉编译工具链，默认路径如下：
 ```
set(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
 ```
 如果交叉编译工具链地址变动，需同步修改CMakeLists.txt
 ## 编译命令
版本包提供了编译脚本build.sh。 AI-Express是支持X2与X3平台，编译的时候需要指定平台信息。具体编译如下   
X3版本编译：
 ```
bash build.sh x3
 ```
X2版本编译：
 ```
bash build.sh x2
 ```
编译的可执行文件和库在build/bin和build/lib目录下

# 部署
版本包提供了部署脚本deploy.sh，可将模型、可执行程序、库文件、配置文件以及测试图片整理到deploy目录中。
 ```
bash deploy.sh
 ```
该脚本会创建deploy部署包，包括如下几个部分：

| 名称             |             备注 |
| ---------------- | ---------------: |
| lib              |       动态依赖库 |
| models           |         模型集合 |
| face_solution    |     人脸解决方案 |
| body_solution    |     人体解决方案 |
| face_body_multisource    |     多路输入多workflow解决方案 |
| ssd_test         | ssd检测模型示例程序 |
| configs          |     vio 配置文件 |
| run.sh           |         运行脚本 |

将deploy目录拷贝到X2/X3开发板上就可以运行参考示例。

此外，版本包中也提供了单元测试的部署脚本deploy_ut.sh，在Linux开发机上执行deploy_ut.sh
```bash
deploy_ut.sh
```
可将单元测试的可执行程序、解决方案可执行程序及模型、配置文件拷贝到deploy目录下。

单元测试位于deploy/unit_test目录下，在开发板执行run_ut.sh即可运行单元测试脚本。
```bash
cd deploy/unit_test
sh run_ut.sh
```
单元测试脚本包含人脸、人体等解决方案，以及每个模块的单元测试程序。

# 运行
直接在开发板的deploy目录下，运行run.sh脚本即可运行指定的测试程序。根据提示，选择对应的序号即可。
各个测试程序的介绍及运行方法请参考相应源码目录下的README.md。    
运行时，只需要在x2/x3设备上运行run.sh脚本，程序跑起来后，在PC上打开Chrome/Edge浏览器，输入x2/x3设备的ip,即可查看展示效果。 

run.sh脚本运行的流程：

第1步，选择需要运行的参考示例的序号;

第2步，选择设备的硬件平台，目前支持X2平台(x2_96board、x2_2610), X3平台(x3_sdb、x3_dev、j3_dev，j3_dev目前只适用于apa示例);

第3步，选择camera组合方式，目前支持single cam(单路sensor)，single feedback(单路回灌)共2种方式，其他方式暂不支持。

第4步，选择开发板上的sensor型号和图像尺寸。其中imx327、os8a10、s5kgm1sp是通过mipi接口与开发板相连，usb_cam是usb摄像头通过USB Type A与开发板相连。

以运行body(人体结构化)为例，具体运行命令：
```
sh run.sh w  # w表示日志为warning等级，也可以输入d(debug)或者i(info)的不同等级，如果不输入日志等级，默认为i等级。
# 然后输入3，选择body，会看到打印 You choose 3:body
# 然后输入3，选择x3_sdb，会看到打印 You choose 3:x3sdb
# 然后输入1，选择single cam,会看到打印 You choose 1:single cam
# 最后输入6，选择single camera: usb_cam, default 1080p，设备上的程序就会开始运行。
```


| 参考示例     |  硬件平台     | 是否支持camera | 是否支持回灌 |  展示客户端  |
| ----------  | ------------ | -----------  | ---------  |  --------  |
| face(人脸抓拍)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| face(人脸抓拍)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可|
| face(人脸抓拍)     |  x3dev |  支持| 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| face(人脸抓拍)     |  x3sdb |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| face_recog(人脸特征提取)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| face_recog(人脸特征提取)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可|
| face_recog(人脸特征提取)     |  x3dev |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| face_recog(人脸特征提取)     |  x3sdb |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| body(人体结构化)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| body(人体结构化)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可|
| body(人体结构化)     |  x3dev |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| body(人体结构化)     |  x3sdb |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| behavior(行为分析)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| behavior(行为分析)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可|
| behavior(行为分析)     |  x3dev |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| behavior(行为分析)     |  x3sdb |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| video_box(多路盒子)     |  x2 96board |  不支持 | 不支持 | 不支持 |
| video_box(多路盒子)     |  x2 面板机 |  不支持 | 不支持 | 不支持 |
| video_box(多路盒子)     |  x3dev |  支持  | 不支持 | HDMI显示器观看 |
| video_box(多路盒子)     |  x3sdb |  支持  | 不支持 | HDMI显示器观看 |
| gesture(手势识别)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| gesture(手势识别)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可|
| gesture(手势识别)     |  x3dev |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| gesture(手势识别)     |  x3sdb |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| xbox(体感游戏)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，可以查看智能结果；http://IP/CrappyBird 与 http://IP/PandaRun 打开游戏界面 |
| xbox(体感游戏)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，可以查看智能结果；http://IP/CrappyBird 与 http://IP/PandaRun 打开游戏界面|
| xbox(体感游戏)     |  x3dev |  支持 | 不支持 | 浏览器输入XJ3的IP地址，可以查看智能结果；http://IP/CrappyBird 与 http://IP/PandaRun 打开游戏界面 |
| xbox(体感游戏)     |  x3sdb |  支持 | 不支持 | 浏览器输入XJ3的IP地址，可以查看智能结果；http://IP/CrappyBird 与 http://IP/PandaRun 打开游戏界面 |
| face_body_multisource(多路输入)     |  x2 96board |  不支持 | 支持 | 不支持 |
| face_body_multisource(多路输入)     |  x2 面板机 |  不支持 | 支持 | 不支持 |
| face_body_multisource(多路输入)     |  x3dev |  不支持 | 支持 | 不支持 |
| face_body_multisource(多路输入)     |  x3sdb |  不支持 | 支持 | 不支持 |
| tv_dance_960x544(智慧电视)     |  x2 96board |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| tv_dance_960x544(智慧电视)     |  x2 面板机 |  支持 | 不支持 | 浏览器输入XJ3的IP地址，点击预览即可|
| tv_dance_960x544(智慧电视)     |  x3dev |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |
| tv_dance_960x544(智慧电视)     |  x3sdb |  支持 | 支持 | 浏览器输入XJ3的IP地址，点击预览即可 |

 ## 硬件说明
| 开发板           |             备注                            |
| --------------  | ---------------:                            |
| 96board         | X2 96board开发板，demo中只配置了1080P的sensor  |
| 2610            | X2 2610 原型机，demo中只配置了1080P的sensor    |
| x3dev           | X3 开发板(大板子)，demo中适配了四种sensor，分别为imx327（1080P），os8a10（2160P），s5kgm（4000x3000）, s5kgm_2160p(2160P)， usb_cam(1080P)|
| x3sdb           | X3 开发板(小板子)，demo中适配了两种sensor：usb_cam(1080P)、os8a10（2160P)|

 ## X3 Sensor说明
 当前AI-Express X3版本适配了四款sensor，分别是imx327， os8a10， s5kgm以及一款usb camera。运行X3示例前一定要确定自己sensor的类别.  
 X2面板机以及96board，sensor的分辨率均为1080P。
| sensor类型      |     图像分辨率       |   备注   |
| ----------     |     ------------    | ------  |
|imx327	| 1920 x 1080  | |
|os8a10 | 3840 x 2160 | |
|s5kgm_2160p| 3840 x 2160| |
|s5kgm  | 4000 x 3000 | 只有body、behavior、xbox这三个solution支持4000x3000分辨率输入  |
|usb_cam| 1920 x 1080 | 目前在x3小板子上测试通过，还会继续优化，usb_camera型号参考https://developer.horizon.ai/forum/id=5f312d96cc8b1e59c8581511

## 回灌方式说明
X3版本回灌支持3种模式，分别为cache， jpg与nv12这3个模式，区别如下：
| 回灌方式      |     功能说明       |   回灌图像配置   |
| ----------     |     ------------    | ------  |
|cache|预先将所有的jpg图像解码到内存中，回灌的时候不需要再进行图像解码操|图片列表配置在：configs/vio_config.json.x3dev.iot.hg中image_list字段|
|jpg|依次读取图像，解码，回灌。若使用循环回灌方式，则每次回灌会单独读取图像解码一次|图片列表配置在：configs/vio_config.json.x3dev.iot.hg中配置的file_path，默认为configs/vio_hg/name_jpg.list|
|nv12|和jpg的区别是回灌的图片是nv12的，只需要读取图像数据，不需要解码|图片列表配置在：configs/vio_config.json.x3dev。iot.hg中配置的file_path，默认为configs/vio_hg/name_nv12.list|

# AI社区相关资源
## 镜像烧录
X3小板子镜像烧录需要的工具以及烧录过程可以参考：https://developer.horizon.ai/forum/id=5f1aa3ee86cc4d95e81a73e6

## x3开发板接入usb camera
X3 接入usb camera，代替mipi camera。具体使用可以参考：https://developer.horizon.ai/forum/id=5f312d96cc8b1e59c8581511

## 多路盒子video_box
多路盒子的solution，具体描述可以参考：https://developer.horizon.ai/forum/id=5f2be161740aaf0beb31234a

## 行为分析behavior
行为分析solution，提供了摔倒检测的功能，功能搭建可以参考：https://developer.horizon.ai/forum/id=5efab48f38ca27ba028078dd

## 体感游戏
可以参考：https://developer.horizon.ai/forum/id=5ef05b412ab6590143c15d6a

## 手势识别
可以参考：https://developer.horizon.ai/forum/id=5f30f806bec8bc98cb72b288

## UVC Device
将X3作为UVC设备，通过USB接口接入android系统的硬件上，x3开发板通过uvc协议传输图像，通过HID协议传输智能结果。具体可以参考： https://developer.horizon.ai/forum/id=5f312a94cc8b1e59c858150c






