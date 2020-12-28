## XProto示例
本节将更详细地介绍Xproto框架中消息的管理。我们仍然基于简单的示例来说明，示例中包括两类消息和两个插件：一个插件负责产生数据并发布，另个插件订阅数据并将接收到的数据进行累加。为了让整个示例过程更清晰，我们在每次消息发布时都会输出日志。

### 定义消息类型
首先需要定义不同插件间传递消息的类型，下面我们创建示例中需要的两类消息并注册到总线。

```c++
#define TYPE_NUMBER_MESSAGE1 "XPLUGIN_NUMBER_MESSAGE1"  // 定义消息类型
#define TYPE_NUMBER_MESSAGE2 "XPLUGIN_NUMBER_MESSAGE2"  // 定义消息类型
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_NUMBER_MESSAGE1)      // 注册消息
XPLUGIN_REGISTER_MSG_TYPE(XPLUGIN_NUMBER_MESSAGE2)      // 注册消息

struct NumberProdMessage1 : horizon::vision::xproto::XProtoMessage {
  float num_;
  explicit NumberProdMessage1(float num) :num_(num) {
    type_ = TYPE_NUMBER_MESSAGE1;     // 指定消息类型
  }
  std::string Serialize() override {
    std::ostringstream ss;
    ss << num_;
    return std::string(ss.str());
  }
};
struct NumberProdMessage2 : horizon::vision::xproto::XProtoMessage {
  float num_;
  explicit NumberProdMessage2(float num) :num_(num) {
    type_ = TYPE_NUMBER_MESSAGE2;     // 指定消息类型
  }
  std::string Serialize() override {
    std::ostringstream ss;
    ss << num_;
    return std::string(ss.str());
  }
};
```

### 定义插件
本示例中，我们同样定义两个插件分别实现消息的发布和订阅。与上个示例有所不同的是，两个插件的涉及到两类消息传递，其中NumberProducerPlugin负责生产两类消息并发布，SumConsumerPlugin负责订阅两类消息并累加。

这里介绍下发布消息的接口`XPlugin::PushMsg()`和`XPlugin::TryPushMsg()`。两个接口都负责向总线发布消息，但**使用PushMsg()发布消息，若订阅该类消息插件的消息队列已达到最大限制，则持续等待直到所有插件的消息队列满足要求，再向总线发布消息；TryPushMsg()，若订阅该类消息的所有插件队列都未满，则发布消息，否则不再发布**。

示例中，NumberProducerPlugin中使用PushMsg()发布NumberProdMessage1类型消息，使用TryPushMsg()发布NumberProdMessage2类型消息，两类消息都最多发布5次：

```c++
  int NumberProducerPlugin::Start() {
    LOGI << "total_cnt=" << total_cnt_;    // total_cnt_ = 5
    LOGI << desc() << " Start";
    prd_thread_ = new std::thread([&] (){
      for (uint32_t i = 0; i < total_cnt_ && !prd_stop_; i++) {
        auto np_msg1 = std::make_shared<NumberProdMessage1>(1);
        // 向总线发送消息，若超出最大限制数量，则持续等待直到消息队列长度满足要求再发送
        PushMsg(np_msg1);
        LOGD << "PushMsg NumberProdMessage1 success";

        auto np_msg2 = std::make_shared<NumberProdMessage2>(1);
        // 向总线发送消息【可能失败】，若未超出最大限制数量，发送消息，返回成功；否则不再发送，返回失败
        int ret = TryPushMsg(np_msg2);
        if (ret == 0) {
          LOGD << "TryPushMsg NumberProdMessage2 success.";
        } else {
          LOGW << "TryPushMsg NumberProdMessage2 fail.";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
    return 0;
  }
```

SumConsumerPlugin插件订阅两类消息，并实现累加功能。为了演示消息队列达到最大限制的情况，这里我们在订阅消息的回调函数中人为休眠一段时间，使得"发布消息速度"大于"订阅消息速度"。
```c++
  int SumConsumerPlugin::Init() {
    sum_ = 0.f;
    RegisterMsg(TYPE_NUMBER_MESSAGE1, std::bind(&SumConsumerPlugin::Sum1,
                                               this, std::placeholders::_1));
    RegisterMsg(TYPE_NUMBER_MESSAGE2, std::bind(&SumConsumerPlugin::Sum2,
                                               this, std::placeholders::_1));
    return XPluginAsync::Init();
  }
  int SumConsumerPlugin::Sum1(XProtoMessagePtr msg) {
    auto np_msg = std::static_pointer_cast<NumberProdMessage1>(msg);
    sum_ += np_msg->num_;
    LOGI << "Consume NumberProdMessage1, curr sum:" << sum_;

    std::this_thread::sleep_for(std::chrono::microseconds(500));
    return sum_;
  }
  int SumConsumerPlugin::Sum2(XProtoMessagePtr msg) {
    auto np_msg = std::static_pointer_cast<NumberProdMessage2>(msg);
    sum_ += np_msg->num_;
    LOGI << "Consume NumberProdMessage2, curr sum:" << sum_;

    std::this_thread::sleep_for(std::chrono::microseconds(500));
    return sum_;
  }
```

### 运行程序
现在实例化两个插件，默认插件的消息队列最大长度是30，我们通过接口`XPluginAsync::SetPluginMsgLimit()`设置SumConsumerPlugin插件的消息队列长度为5。下面顺序执行Init()、Start()、Stop()、DeInit()来运行程序。

```c++
  SetLogLevel(HOBOT_LOG_DEBUG);

  auto np_plg = std::make_shared<NumberProducerPlugin>();
  auto sc_plg = std::make_shared<SumConsumerPlugin>();

  // 设置该插件的最大消息队列大小，默认30
  sc_plg->SetPluginMsgLimit(5);

  np_plg->Init();
  sc_plg->Init();
  np_plg->Start();
  sc_plg->Start();

  std::this_thread::sleep_for(std::chrono::seconds(5));

  np_plg->Stop();
  sc_plg->Stop();
  np_plg->DeInit();
  sc_plg->DeInit();
```

来看下运行程序的输出结果：
```
(plugins.h:34): total_cnt=5
(plugins.h:35): NumberProducerPlugin Start
(plugins.h:103): SumConsumerPlugin Start

(plugins.h:41): PushMsg NumberProdMessage1 success        // 消息队列长度1
(plugins.h:47): TryPushMsg NumberProdMessage2 success.    // 消息队列长度2
(plugins.h:41): PushMsg NumberProdMessage1 success        // 消息队列长度3
(plugins.h:47): TryPushMsg NumberProdMessage2 success.    // 消息队列长度4
(plugins.h:88): Consume NumberProdMessage1, curr sum:1    // 消息队列长度3
(plugins.h:41): PushMsg NumberProdMessage1 success        // 消息队列长度4
(plugins.h:47): TryPushMsg NumberProdMessage2 success.    // 消息队列长度5
// 队列已满，PushMsg等待队列长度减少再发送
(msg_manager.h:100): PushMsg type: XPLUGIN_NUMBER_MESSAGE1, try to wait 500us until plugin msg-queue is free.
(plugins.h:96): Consume NumberProdMessage2, curr sum:2    // 消息队列长度4
(plugins.h:41): PushMsg NumberProdMessage1 success        // 消息队列长度5
// 队列已满，TryPushMsg发布消息失败
(plugins.h:49): TryPushMsg NumberProdMessage2 fail.       // 消息队列长度5
// 队列已满，PushMsg等待队列长度减少再发送
(msg_manager.h:100): PushMsg type: XPLUGIN_NUMBER_MESSAGE1, try to wait 500us until plugin msg-queue is free.
(plugins.h:88): Consume NumberProdMessage1, curr sum:3    // 消息队列长度4
(plugins.h:41): PushMsg NumberProdMessage1 success        // 消息队列长度5
// 队列已满，TryPushMsg发布消息失败
(plugins.h:49): TryPushMsg NumberProdMessage2 fail.       // 消息队列长度5
(plugins.h:96): Consume NumberProdMessage2, curr sum:4    // 消息队列长度4
(plugins.h:88): Consume NumberProdMessage1, curr sum:5    // 消息队列长度3
(plugins.h:96): Consume NumberProdMessage2, curr sum:6    // 消息队列长度2
(plugins.h:88): Consume NumberProdMessage1, curr sum:7    // 消息队列长度1
(plugins.h:88): Consume NumberProdMessage1, curr sum:8    // 消息队列长度0
(plugins.h:62): NumberProducerPlugin Stop
(plugins.h:107): SumConsumerPlugin Stop
```
