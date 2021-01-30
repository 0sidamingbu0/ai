# tutorials

本目录提供如何使用xproto的示例代码。

# 编译

## 编译tutorials(CentOS平台)

编译命令如下：

```bash
cd output/xproto/tutorials/ && mkdir build && cd build
cmake .. -DX86_ARCH=ON -DX86_CENTOS=ON 
make -j
```

## 编译tutorials(Ubuntu平台)

编译命令如下：

```bash
cd output/xproto/tutorials/ && mkdir build &&  cd build
cmake .. -DX86_ARCH=ON
make -j
```

## 编译tutorials(aarch64平台)

编译命令如下：

```
cd output/xproto/tutorials/ && mkdir build && cd build 
cmake .. 
make -j
```

编译结束后，生成的可执行文件及运行所需配置文件在xproto/tutorials/build下。

xproto库的示例使用教程可参考xproto/tutorials/下每个stage的说明文档。

编译xproto库的tutorials方法，与xstream相同。
