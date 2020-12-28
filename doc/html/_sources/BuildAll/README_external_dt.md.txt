车路协同代码包
=======

# 介绍

本文档对发布物的内容、编译、运行做简要介绍。

# 发布物说明
## 目录结构

发布物顶层目录结构如下：

![vehicle_solution代码包目录结构](./doc/image/vehicle_solution_tree.png)

| 名称 | 备注 |
| --- | --- |
| build.sh | 编译脚本 |
| CMakeLists.txt | CMake脚本 |
| deploy.sh | 打包脚本 |
| deps | 依赖项 |
| models | 模型文件 |
| README.md | 说明文档 |
| run.sh | 运行脚本 |
| source | 程序源代码 |
| start_nginx.sh | nginx服务启动脚本 |
| webservice | web服务源代码 |

程序源代码目录结构如下：

![vehilce_solution_source目录结构](./doc/image/vehicle_solution_source_tree.png)

| 名称 | 备注 |
| --- | --- |
| common/xproto/framework | XProto框架代码 |
| common/xproto/msgtype | XProto框架中消息类型基类代码 |
| common/xproto/plugins | XProto框架中插件代码 |
| common/xstream/framework | XStream框架代码 |
| common/xstream/vision_type | 计算机视觉任务常用数据结构代码 |
| solution_zoo/vehicle | 应用程序代码 |
| solution_zoo/xstream/methods | XStream Methods代码 |

## 代码模块

您可以参考发布物中的代码进行模型集成和智能结果的解析。  
本Solution中包含多任务检测模型、车牌识别模型、车型分类模型和车颜色分类模型。  
代码中使用*DnnPredictMethod*类(source/common/xstream/framework/methods/DnnPredictMethod)和*DnnPostProcessMethod*类(source/common/xstream/framework/methods/DnnPostProcessMethod)进行模型集成。它们的主要接口如下：  

**DnnPredictMethod**类
| 接口 | 说明 |
| --- | --- |
| AllocInputTensor() | 申请InputTensor大小 |
| AllocOutputTensor() | 申请OutputTensor大小 |
| PrepareInputData() | 派生类实现，用于预处理 |
| GetSrcImageSize() | 获取输入图像大小 |
| DoProcess() | 主逻辑，完全复用 |

**DnnPostProcessMethod**类
| 接口 | 说明 |
| --- | --- |
| ParseDnnResult() | 派生类实现，用于后处理 |
| DoProcess() | 主逻辑，完全复用 |

对于多任务检测模型，您可以参考*MultitaskPredictMethod*类(source/solution_zoo/xstream/methods/MultitaskPredictMethod)和*MultitaskPostProcessMethod*类(source/solution_zoo/xstream/methods/MultitaskPostProcessMethod)进行集成。其中主要接口如下：

**MultitaskPredictMethod类**:  
| 接口 | 说明 |
| --- | --- |
| GetSrcImageSize() | 派生类实现 |
| PrepareInputData() | 派生类实现 |

**MultitaskPostProcessMethod类**:  
| 接口 | 说明 |
| --- | --- |
| ParseDnnResult() | 派生类实现 |
| GetRppRects() | 解析BPU输出数据得到检测框 |
| GetKps() | 解析BPU输出数据得到车辆关键点 |
| CoordinateTransform() | 坐标转换 |

对于其他模型，您可以参考*RectInputPredictMethod*类(source/solution_zoo/xstream/methods/RectInputPredictMethod)和*VehicleTyepPostProcessMethod*类(source/solution_zoo/xstream/methods/VehicleTypePostProcessMethod)进行集成。其中主要接口如下：

**RectInputPredictMethod类**:  
| 接口 | 说明 |
| --- | --- |
| PrepareInputData() | 派生类实现 |

**VehicleTypePostProcessMethod类**:  
| 接口 | 说明 |
| --- | --- |
| ParseDnnResult() | 派生类实现 |
| ParseVehicleType() | 解析BPU输出数据得到车型 |

# 编译
## 编译环境

需提前准备好交叉编译工具链，默认路径如下：
```
set(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
```
如果交叉编译工具链地址变动，需同步修改CMakeLists.txt

## 编译命令

执行如下命令，编译的可执行文件和库在build/bin和build/lib目录下。hbipc参数仅限x2使用HbipcPlugin作为输出plugin时使用。
```
bash build.sh < x2 > [ hbipc ]
```

# 部署

```
bash deploy.sh
```
该脚本会创建deploy部署包，包括如下几个部分：

| 名称             |             备注 |
| ---------------- | ---------------: |
| lib              |       动态依赖库 |
| models           |         模型集合 |
| vehicle_solution |     车辆解决方案 |
| configs          |     vio 配置文件 |
| run.sh           |         运行脚本 |

# 运行

直接运行run.sh脚本即可运行指定的测试程序。默认使用96baord配置。各个测试程序的介绍及运行方法请参考相应源码目录下的README.md。  
 ```
sh run.sh vehicle < 96board | x2-ipc > < 720p | 1080p > [ d | i | w | e | f ]
 ```
 ## 硬件说明
| 开发板           |             备注                            |
| --------------  | ---------------:                            |
| 96board         | X2 96board开发板，demo中只配置了1080P的sensor，720p目前配置为回灌  |
| x2-ipc          | X2 IPC, demo中配置了1080P的sensor。用于演示BT1120输入SPI输出。     |

