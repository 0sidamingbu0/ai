# 快速上手

AI Express，中文名称AI应用开发中间件，是地平线芯片“天工开物”（Horizon OpenExplorer™️ Platform）AI开发平台的一部分，旨在通过全面降低开发者门槛、提升开发速度、保证开发质量，赋能产业智慧升级。

## 概述

概述部分将一步一步介绍如何搭建AI Express开发环境、AIExpress源码如何下载和编译部署、如何运行AIEXPRESS自带的人体结构化参考方案(body solution)、如何运行测试用例、以及如何生成xstream和xproto库，AIExpress用户手册和地平线开发者社区相关资源。

## 1.搭建AI Express开发环境

搭建开发环境，主要包括：硬件准备、搭建软件开发环境和搭建开发板环境。

### 相关组件版本的匹配关系
| AIExpress版本        |    浮点转换工具链版本  |  系统镜像版本         |
| ---------------- | --------------- | --------------- |
| 2.10.0           |  1.1.21  |       20210207         |
| 2.9.0            |  1.1.20  |       20210104         |
| 2.8.0            |  1.1.19  |       20201124         |
| 2.7.0            |  1.1.18  |       20201023         |
| 2.6.0            |  1.1.16  |       20200922         |

### 硬件准备

* 1台安装64位Linux操作系统的开发机(或者虚拟机)。支持的操作系统版本Ubuntu、CentOS，主要用于编译AIExpress代码和日常开发。
* 1台安装Windows操作系统的开发机，用于烧录旭日3开发板系统镜像，串口调试。
* 1个1080p的USB摄像头
* 1块x3sdb开发板(旭日3 AI 开发板)、1根USB转串口下载线、1根Micro USB线

### 搭建软件开发环境

**a. Linux开发机环境准备**

* 本地安装

 1).安装`CMake 3.15+`以上版本。安装方式如下：

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz \
    && tar -zxvf cmake-3.17.2.tar.gz \
    && cd cmake-3.17.2 \
    && ./bootstrap \
    && make \
    && sudo make install \
    && cd .. \
    && rm -rf cmake-3.17* 
```
2).安装交叉编译工具链。

下载并安装芯片交叉编译工具[gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu](https://pan.horizon.ai/index.php/s/d3QH3MfzHT5fwd2)，推荐安装路径:`/opt/`，如果交叉编译工具链有更新，需同步修改工具链配置。具体文件：AIEXPRESS代码工程根目录下的CMakeLists.txt, source/common/xstream/framework/CMakeLists.txt和source/common/xproto/framework/CMakeLists.txt。
具体修改内容：

```bash
set(CMAKE_C_COMPILER /opt/${工具链目录名}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/${工具链目录名}/bin/aarch64-linux-gnu-g++)
```

* 使用Docker环境

获取镜像文件及使用说明:  https://pan.horizon.ai/index.php/s/md6mbFsei4DWMJq

**b. Windows开发机环境准备**

* 下载开发板系统镜像及烧录工具：[开发板系统镜像及烧录工具下载地址](https://developer.horizon.ai/forum/id=5f156192740aaf0beb3119dd)

各文件说明如下：

**注意：该文档列的工具和镜像版本未保证是最新版本，最新版本请联系地平线获取**：

  | 名称             |             说明 |
| ---------------- | ---------------: |
| disk_X3SDB-Linux-20201214wb_2G.img              |       x3sdb开发板系统镜像文件 |
| hbupdate_win64_0.7.6.zip           |         hbupdate开发板烧录工具(Windows) |
| win32diskimager-1.0.0-install.zip    |     win32diskimager开发板烧录工具 |
| CP210x_USB2UART_Driver.zip    |     Windows串口驱动安装包 |
| PL2302-USB-to-Serial Comm Port.zip    |     Windows串口驱动安装包 |
| PL2303-M_LogoDriver_Setup_v202_20200527.zip         | Windows串口驱动安装包 |
| hbupdate_linux_0.7.6.tgz           |         hbupdate开发板烧录工具(Linux) |

### 搭建开发板环境

#### 开发板接线

* 将USB Camera插入板子的USB口中，将网线一端插入开发板网口，另一端插入Windows开发机，电源插入开发板电源接口。
* 目前x3sdb开发板适配了1080p的USB摄像头([具体型号参考](https://developer.horizon.ai/forum/id=5f312d96cc8b1e59c8581511))

#### 升级开发板系统镜像版本

* 开发板系统镜像烧录教程: [X3开发板板镜像烧录教程](https://developer.horizon.ai/forum/id=5f890b0ccc8b1e59c8584bfc)

* 开发板使用注意事项：https://developer.horizon.ai/forum/id=5efac2d32ab6590143c16024

更多参考：[x3sdb开发板资料包（最新汇总版）](https://developer.horizon.ai/forum/id=5f156192740aaf0beb3119dd)

## 2.AIExpress源码编译与部署运行

### AIExpress源码下载

AIExpress github地址:https://github.com/HorizonRobotics-Platform/AI-EXPRESS

```bash
git clone git@github.com:HorizonRobotics-Platform/AI-EXPRESS.git
```

### 编译

代码仓库提供了编译一键脚本build.sh，git clone代码后可直接编译。 编译时需要指定平台信息即可，具体编译如下：

* 本地环境

本地环境可以直接编译:

```bash
cd AI-EXPRESS
bash build.sh x3
```

* Docker环境

登录docker环境，执行如下命令进行编译:

```bash
cd AI-EXPRESS
bash build.sh x3
```

编译的可执行文件和库在build/bin和build/lib目录下

### 部署

代码仓库提供了一键部署脚本deploy.sh，可将模型、可执行程序、库文件、配置文件以及测试图片整理到deploy部署目录中。ssh登录到开发板，通过scp命令，将deploy目录拷贝到x3sdb开发板上就可以运行人体结构化参考方案。

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
| video_box        |     视频盒子解决方案 |
| configs          |     vio 配置文件 |
| run.sh           |         运行脚本 |

### 运行参考方案

直接在开发板的deploy目录下，运行run.sh脚本即可运行指定的测试程序。以人体结构化参考方案为例，具体运行命令：

**注意：具体示例的选择，参考可执行程序的实际打印，该文档更新可能滞后**
```
sh run.sh w  # w表示日志为warning等级，也可以输入d(debug)或者i(info)的不同等级
# 然后输入3，选择body，会看到打印 You choose 3:body
# 然后输入3，选择x3_sdb，会看到打印 You choose 3:x3sdb
# 然后输入1，选择single cam,会看到打印 You choose 1:single cam
# 最后输入6，选择single camera: usb_cam, default 1080p
```

### 结果展示

当开发板上run.sh程序执行后，可以在PC上打开Chrome浏览器，输入x3sdb开发板的ip，然后点击页面的[Web展示端]，即可查看人体结构化解决方案(body solution)的效果。  

各个测试程序的介绍及运行方法请参考**场景参考解决方案**或者**常见问题**章节。

除了上述人体结构化解决方案外，人脸结构化参考方案，人脸识别参考方案和视频盒子参考方案也已开源。AIExpress会陆续开源**人体行为分析参考方案**,**体感游戏参考方案**,**apa自动泊车参考方案**，有任何建议或问题，欢迎在github上提Issue。


## 3.运行测试用例

### 生成测试用例的部署包

对AIExpress源码编译后，在Linux开发机上执行deploy_ut.sh脚本，生成含有测试用例的部署目录，并将单元测试的可执行程序、解决方案可执行程序及模型、配置文件拷贝到deploy目录下。
```bash
bash deploy_ut.sh
```
ssh登录到开发板，通过scp命令，将deploy目录拷贝到x3sdb开发板上。

### 运行测试用例

单元测试程序位于deploy/unit_test目录下，在开发板上执行run_ut.sh脚本，即可运行测试用例。
```bash
cd deploy/unit_test
sh run_ut.sh
```
单元测试脚本暂时只支持face(人脸结构化参考方案)、body(人体结构化参考方案)、video_box(视频盒子参考方案)，以及aiexpress每个模块的单元测试程序。

## 4.独立编译生成xstream和xproto库

AI-EXPRESS支持独立编译生成xstream和xproto库，目前支持aarch64(默认)/Ubuntu/CentOS，共3种平台。可根据自己具体的开发环境来选择对应的平台。

* CentOS平台(CentOS7, gcc4.8.5)

```bash
cd AI-EXPRESS/source/common/xstream/framework/
mkdir build && cd build
cmake .. -DX86_ARCH=ON -DX86_CENTOS=ON 
make -j && make install
```

* Ubuntu平台(Ubuntu18.04, gcc7.5.0)

```bash
cd AI-EXPRESS/source/common/xstream/framework/
mkdir build &&  cd build
cmake .. -DX86_ARCH=ON
make -j && make install
```

* Linaro-aarch64平台(gcc-linaro-6.5-2018.12)

```bash
cd AI-EXPRESS/source/common/xstream/framework/
mkdir build &&  cd build
cmake ..
make -j && make install
```

默认编译xstream会生成libxstream.a，如果需要生成libxstream.so，可通过修改AI-EXPRESS/source/common/xstream/framework/CMakeLists.txt中的编译选项`BUILD_SHARED_LIBS`为`true`进行编译：

```bash
set(BUILD_SHARED_LIBS true)
```

或者在`cmake ..`时，添加-DBUILD_SHARED_LIBS=ON选项，即可。


编译结束后，在AI-EXPRESS/source/common/xstream/framework下会生成output目录，output目录包含libxstream.a、头文件和框架说明文档、入门教程文档和教程代码，支持独立使用xstream库。

编译xproto库方法，与xstream相同，即在AI-EXPRESS/source/common/xproto/framework下创建build目录，根据需要，添加不同的CMake编译选项，进行编译即可。

## 5.AIExpress框架文档

参考doc目录下导航页`doc/html/index.html`。


## 6.[地平线开发者社区相关资源](https://developer.horizon.ai/)
### 镜像烧录
参考 https://developer.horizon.ai/forum/id=5f890b0ccc8b1e59c8584bfc

### 多路盒子video_box

多路盒子的solution，具体描述可以参考：https://developer.horizon.ai/forum/id=5f2be161740aaf0beb31234a

### 行为分析behavior

行为分析solution，提供了摔倒检测的功能，功能搭建可以参考：https://developer.horizon.ai/forum/id=5efab48f38ca27ba028078dd

此外提供了手势识别的参考方案，可以参考https://developer.horizon.ai/forum/id=5f30f806bec8bc98cb72b288

### 体感游戏

可以参考：https://developer.horizon.ai/forum/id=5ef05b412ab6590143c15d6a

### usb camera智慧电视

将X3作为UVC设备，通过USB接口接入android系统的硬件上，x3sdb开发板通过uvc协议传输图像，通过HID协议传输智能结果。具体可以参考： https://developer.horizon.ai/forum/id=5f312a94cc8b1e59c858150c

### 手势识别
参考：https://developer.horizon.ai/forum/id=5f30f806bec8bc98cb72b288

### X3接入外接usb camera
参考：https://developer.horizon.ai/forum/id=5f312d96cc8b1e59c8581511

### efuse烧录
参考：https://developer.horizon.ai/forum/id=5f881ceccc8b1e59c8584b6d
