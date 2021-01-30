# usb cam sdk使用说明

## 介绍
sdk部署包中包括已编译好的可执行程序、依赖库、配置文件、模型、sdk的头文件、sample程序和编译脚本。

## 编译和运行
1、直接运行release包中的可执行程序
	release包中包含可执行程序，执行脚本为start_sdk.sh，直接运行脚本，根据提示选择运行的功能。
	（1）start_sdk.sh cam	#os8a10 sensor作为输入
	（2）start_sdk.sh fb	    #本地图片作为输入

2、编译sample源程序并运行
	release包中包含sdk的参考sample源程序和编译脚本，支持客户使用sdk做方案开发。
	sample程序的路径为release/sdk_sample，使用sample编译和运行方法：
	cd release/sdk_sample	#进入sample路径
	sh build.sh				#编译和部署
	sh start_sdk.sh			#运行

