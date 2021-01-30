# <font color=#0000ff>V2.9.0</font>

## Release Date
2021/01/31

## Features

- X3对齐系统软件月度对外发布的版本20210104版本
- bpu-predict接口以及hbdk对齐浮点工具链1.1.20版本
- 新增单独的脚本运行人脸、人体、视频盒子、智慧电视、YoloV3示例
- 对vehicle与face_body_multisource代码做了重构，更清晰。
- 修复视频盒子拉取码流失败后重连crash的问题
- vehicle支持车牌识别、车颜色识别以及2D定位功能
- YoloV3的输入从图像中间区域抠图缩放预测改为基于图像padding后缩放送入模型预测
- 修复methods编译打包后缺少头文件的问题
- XStream-Framework与XProto-Framework 代码行覆盖率90%以上