# 概述
XProto是一个插件管理和消息订阅分发框架.它将插件共有的特性抽象成一个标准基类,使用XProto的插件必须继承于该基类.  
基于XProto的Plugin是通过消息驱动的, 所有的Plugin都挂载到XProto的内部消息总线中,当一个插件产生消息并把消息Push到总线之后,其他订阅该消息的插件回调函数就会被调用.每一个Plugin都可以向总线订阅和发布消息.  
XProto与Plugin之间的关系如下图:  

![XProto与Plugin之间的关系](document/img/relationship_between_xproto_and_plugin.png "XProto与Plugin之间的关系")

# XProto设计理念
XStream聚集算法模型、策略的集成以及最终业务Workflow/SDK生成。而XProto是在XStream基础上，为基于XStream构建算法SDK提供APP的运行环境。它支持快速将XStream Workflow封装成可运行的APP，并运行在地平线边缘芯片中。

# XProto组成
XProto主要由XProto-Framework， msgtype， plugins等部分组成。其中XProto-Framework为框架本身内容，msgtype与plugins均是实现参考示例时的实现参考；客户基于XProto开发，可以不依赖msgtype与plugins。
## XProto-Framework
XProto Framework原型框架主要包括Plugin插件管理和消息分发两个部分。Plugin插件是一个任务实体，所有的Plugin插件都连接到XProto消息总线中，当一个Plugin插件产生消息并把消息Push到总线之后，其他订阅该消息的Plugin插件就可以被调用。

XProto Framework简化了用户的开发任务，使其专注实现plugin本身。
XProto原型应用开发框架，定义了基础的消息类型、Plugin基础类定义以及消息发布订阅的处理逻辑。开发者基于XProto-Framework库+基础消息类型+Plugin基础类完成功能开发。
### Msgtype
定义各个组件使用的数据类型和proto的数据结构定义，方便不同的plugin产生的数据在plugins间传输，在已沉淀的solution中应用较多，比如VioMessage、SmartMessage、RtspMessage等。
开发者在使用xproto开发时，可以参考Msgtype；如果是开发者自己开发，可以不依赖Msgtype。
### plugins
xproto实现、沉淀了一些有用的插件供给客户复用，方便客户快速继承app。比如vioplugin用于获取图像输入、Rtspplugin用于把视频流直按照RTSP输出方便实时查看视频、websocketplugin实现了视频和智能数据的同步并支持在web上显示、uvcplugin则实现了uvc协议，方便在智能设备上实时显示视频和智能数据等。
推荐客户复用或扩展已有的plugin，也可以使用xstream实现自己的app框架。
## XProto-Framework用户手册
详细介绍xproto的运行原理及使用方法，方便客户理解xproto后快速构建app。
### 基本概念与运行机制
XProto Framework原型框架主要包括Plugin插件管理和消息分发两个部分。Plugin插件是一个任务实体，所有的Plugin插件都连接到XProto消息总线中，当一个Plubin插件产生消息并把消息Push到总线之后，其他订阅该消息的Plugin插件就可以被调用。

每一个Plugin插件都可以向总线订阅和发布消息，通过消息驱动方式来实现整个原型应用的落地。

### 基础数据结构描述
#### 消息基础类
消息基础类XProtoMessage，新的Message类型需要继承XProtoMessage。
```c++
struct XProtoMessage : public std::enable_shared_from_this<XProtoMessage> {
  XProtoMessage &operator=(const XProtoMessage &) = delete;
  virtual ~XProtoMessage() = default;

  std::string type_ = "NONE";

  std::string param_ = "";

  std::string type() const { return type_; }

  virtual std::string Serialize() = 0;

  virtual void *ConvertData() {
    // need call delete() to release the output
    return nullptr;
  }
};

using XProtoMessagePtr = std::shared_ptr<XProtoMessage>;
```

消息的声明与定义：
* 使用宏XPLUGIN_REGISTER_MSG_TYPE,自定义消息类型，每个消息名字唯一；
* 监听消息的插件需要：
* 实现消息处理函数；
* 覆盖Init函数，在其中完成监听消息注册，并绑定对应的消息处理函数，
* 及其他初始化工作，同时在函数返回前需要调用父plugin的Init方法，
* 通常是XPluginAsync::Init()。

相关接口定义如下：
```c++
// 声明消息类型;每一类消息都有一个字符串形式的消息类型和结构体来表示.
// 该接口为一个宏, 参数MSG_TYPE用来表示声明的消息类型, 需要直接使用标识符的格式书写, 宏内部会将其转成字符串.  
// 注意：需要在消费者Plugin调用订阅消息接口之前调用该接口声明消息类型,一般将该宏放在全局变量声明的位置.
// 参数：MSGTYPE: 消息类型
XPLUGIN_REGISTER_MSG_TYPE(MSG_TYPE)

// 初始化Plugin
// 该接口需要继承XPluginAsync类的自定义Plugin实现该接口定义. 该接口用来初始化Plugin，自定义Plugin一般在该接口内调用订阅消息接口, 然后继续调用XPluginAsync::Init接口以初始化父类.
// 返回值：0: 成功；非0: 失败
int XPluginAsync::Init() override;

// 发布消息
// 该接口用来将消息发布到XProto内部总线上. 接收一个类型为XProtoMessage的结构体指针, XProto的所有消息都继承于该类型.
// 参数：XProtoMessagePtr msg: 发布到总线的消息. 
void XPluginAsync::PushMsg(XProtoMessagePtr msg);

// 订阅消息
// 订阅指定类型的消息. 监听总线, 当指定的消息类型发布时, 调用回调函数.  
// 自定义的Plugin需要在Init函数中，调用XPluginAsync::Init之前调用该接口完成监听消息注册。
// 参数：const std::string& type: 消息类型字符串.
// 参数：XProtoMessageFunc callback: 该类型消息的回调函数.
void XPluginAsync::RegisterMsg(const std::string& type, XProtoMessageFunc callback);
```

## Build
### 准备环境
#### 安装cmake
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz \
    && tar -zxvf cmake-3.17.2.tar.gz \
    && cd cmake-3.17.2 \
    && ./bootstrap \
    && make \
    && sudo make install \
    && cd .. \
    && rm -rf cmake-3.17
```

#### 安装交叉编译工具链
  
  可直接下载：http://releases.linaro.org/components/toolchain/binaries/6.5-2018.12/aarch64-linux-gnu/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu.tar.xz
  或者联系地平线技术支持人员获取: [gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu](https://pan.horizon.ai/index.php/s/d3QH3MfzHT5fwd2)
   
#### 编译

AI-EXPRESS支持独立编译生成xstream和xproto库，目前支持aarch64(默认)/Ubuntu/CentOS，共3种平台。可根据自己具体的开发环境来选择对应的平台。

* CentOS平台

```bash
cd AI-EXPRESS/source/common/xproto/framework/
mkdir build && cd build
cmake .. -DX86_ARCH=ON -DX86_CENTOS=ON 
make -j && make install
```

* Ubuntu平台

```bash
cd AI-EXPRESS/source/common/xproto/framework/
mkdir build &&  cd build
cmake .. -DX86_ARCH=ON
make -j && make install
```

* Linaro-aarch64平台

```bash
cd AI-EXPRESS/source/common/xproto/framework/
mkdir build &&  cd build
cmake ..
make -j && make install
```

默认编译xproto会生成libxproto.a，如果需要生成libxproto.so，可通过修改AI-EXPRESS/source/common/xproto/framework/CMakeLists.txt中的编译选项`BUILD_SHARED_LIBS`为`true`进行编译：

```bash
set(BUILD_SHARED_LIBS true)
```

或者在`cmake ..`时，添加-DBUILD_SHARED_LIBS=ON选项，即可。


编译结束后，在common/xproto/framework下会生成output目录，output目录包含libxproto.a、头文件和框架说明文档、入门教程文档和教程代码，支持独立使用xproto库。

### Plugin接口描述
一个plugin代表一个单独的功能模块，可以向总线订阅和发布消息，通过消息驱动方式来实现整个原型应用的落地。
插件可能会生产消息或者向总线注册监听某类消息。如果生产消息需要调用PushMsg()将消息发送到总线分发;如果监听消息，需要实现消息处理函数，并在Init函数中注册需要监听的消息类型，绑定对应的消息处理函数，同时在Init函数返回前调用父plugin的Init方法，通常是XPluginAsync::Init()。
插件提供了两个基础类XPlugin和XPluginAsync，继承这两个基础类之一可实现一个新的plugin，通过override的函数包括:`Init()、Start()、Stop()、Desc();`等接口来管理插件的生命周期。两个plugin基类功能上有些差异：
* XPlugin类：定义了管理周期的相关接口以及发送接收消息的接口，接收到的消息需要自己管理分发
* XPluginAsync类：在XPlugin的基础上，增加了消息的分发功能和流量管理，用户只需要实现消息回调函数即可
```c++
class XPlugin : public std::enable_shared_from_this<XPlugin> {
 public:
  XPlugin() = default;
  virtual ~XPlugin() = default;
  // 完成register msg和workflow的初始化操作
  virtual int Init() = 0;

  virtual int DeInit() {
    return 0;
  }
  // 处理register的msg类型，如有需要，push自己的msg到总线
  virtual void OnMsg(XProtoMessagePtr msg) = 0;
  // 用于流量管理
  virtual int GetPluginMsgCount() = 0;
  virtual int GetPluginMsgLimit() = 0;
  virtual void SetPluginMsgLimit(int msg_limit_count) = 0;

  virtual std::string desc() const {
    return "XPlugin";
  }

 protected:
  // 向总线注册监听消息类型
  void RegisterMsg(const std::string& type);
  // 卸载监听消息类型
  void UnRegisterMsg(const std::string& type);
  // 向总线推送消息
  void PushMsg(XProtoMessagePtr msg);
};

class XPluginAsync : public XPlugin {
 public:
  XPluginAsync();
  explicit XPluginAsync(int thread_num);
  ~XPluginAsync() = default;
  // 注册监听消息类型到总线+plugin的初始化
  int Init() override;
  int DeInit() override;
  // 消息处理上半部分，将消息推送该plugin的消息队列 + 流量控制
  void OnMsg(XProtoMessagePtr msg);
  int GetPluginMsgCount();
  int GetPluginMsgLimit();
  void SetPluginMsgLimit(int msg_limit_count);
  // 启动Plugin
  virtual int Start() {
    return 0;
  }
  // 停止Plugin
  virtual int Stop() {
    return 0;
  }
#ifdef PYAPI
  void RegMsg(const std::string &type,
              std::function<void(pybind11::object)> cb);
#endif

 protected:
  using XProtoMessageFunc = std::function<int(XProtoMessagePtr)>;
  // 重载注册监听消息类型接口，注册时包含异步回调函数
  // 该函数中需要调用RegisterMsg(XpluginMessageType type)完成总线注册
  // Note: 自定义的plugin需要在Init函数中，
  //       调用XPluginAsync::Init之前调用该接口完成监听消息注册。
  void RegisterMsg(const std::string& type, XProtoMessageFunc callback);

 private:
  // 消息处理下半部分，分发消息并调用对应的callback函数
  void OnMsgDown(XProtoMessagePtr msg);

  hobot::CThreadPool msg_handle_;
  std::mutex msg_mutex_;
  int msg_limit_count_ = 0;
  std::mutex msg_limit_mutex_;
  std::map<std::string, XProtoMessageFunc> msg_map_;
};
```
相关接口描述如下：
```c++
// 初始化Plugin
// 该接口需要继承XPluginAsync类的自定义Plugin实现该接口定义. 该接口用来初始化Plugin，自定义Plugin一般在该接口内调用订阅消息接口, 然后继续调用XPluginAsync::Init接口以初始化父类.
// 返回值：0: 成功；非0: 失败
int XPluginAsync::Init() override;

// 启动Plugin
// 该接口需要继承XPluginAsync类的自定义Plugin实现该接口定义. 该接口用来启动Plugin. 
// 返回值：0: 成功；非0: 失败
int XPluginAsync::Start();

// 停止Plugin
// 该接口需要继承XPluginAsync类的自定义Plugin实现该接口定义. 该接口用来停止Plugin. 
// 返回值：0: 成功；非0: 失败
int XPluginAsync::Stop();

// 反初始化Plugin
// 该接口需要继承XPluginAsync类的自定义Plugin实现该接口定义. 该接口用来反初始化Plugin.  继承自XPluginAsync子plugin类，在完成自己的反初始化任务后，最后需要调用XPluginAsync::DeInit接口以反初始化父类.
// 返回值：0: 成功；非0: 失败
int XPluginAsync::DeInit() override;

// 插件描述信息
// 该接口需要继承XPluginAsync类的自定义Plugin实现该接口定义. 
// 返回值：描述当前自定义Plugin的字符串.
std::string XPluginAsync::desc() const;
```
**接口详细用法参考`sample_plugin.cpp`**

### XProto开发示例
XProto提供了一系列的开发示例，基于XProto实现了简单的功能，介绍`消息`和`插件`的基本使用，和XProto消息管理的高阶功能。

#### stage1 消息的订阅和发布
本节将介绍如何使用Xproto框架实现不同插件之间消息的发布和订阅。
详细参见[tutorials_stage1 消息的订阅和发布](./tutorials/stage1/README.md)

#### stage2 插件的最大消息队列
本节将介绍Xproto框架中消息的管理，有的业务场景下消息产生和消费速率不同，容易造成总线上消息的堆积，本节将介绍如何设置插件的最大消息队列数。

详细参见[tutorials_stage2 插件的最大消息队列](./tutorials/stage2/README.md)

#### stage3 消息处理超时警告
本节介绍使用Xproto框架对消息处理的耗时进行监控。当插件处理消息的耗时超过默认时长，程序内回抛出警告日志。
详细参见[tutorials_stage3 消息处理超时警告](./tutorials/stage3/README.md)
