# tutorials

本目录提供如何使用xstream的示例代码。

# 编译

## 编译tutorials(CentOS平台)

编译命令如下：

```bash
cd output/xstream/tutorials/ && mkdir build && cd build
cmake .. -DX86_ARCH=ON -DX86_CENTOS=ON 
make -j && make install
```

## 编译tutorials(Ubuntu平台)

编译命令如下：

```bash
cd output/xstream/tutorials/ && mkdir build &&  cd build
cmake .. -DX86_ARCH=ON
make -j && make install
```

## 编译tutorials(aarch64平台)

编译命令如下：

```
cd output/xstream/tutorials/ && mkdir build && cd build 
cmake .. 
make -j && make install
```

编译结束后，生成的可执行文件及运行所需配置文件在xstream/tutorials/build下。

xstream库的示例使用教程可参考xstream/tutorials/下每个stage的说明文档。

编译xproto库的tutorials方法，与xstream相同。
