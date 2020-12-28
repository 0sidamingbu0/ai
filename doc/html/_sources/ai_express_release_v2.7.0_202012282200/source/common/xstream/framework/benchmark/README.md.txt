## XStream Framework调度性能评测
本节主要对XStream Framework调度性能进行评测说明。

我们将基于下面的workflow进行调度评测，具体如下。workflow输入数据是input, 输入数据先后经过两个PassThrough节点的运算，最终输出数据是output。其中PassThrough实现了输入输出数据的透传。
```json
{
    "inputs": ["input"],
    "outputs": ["output"],
    "workflow": [
      {
        "method_type": "PassThrough",
        "unique_name": "pass_through_1",
        "inputs": [
          "input"
        ],
        "outputs": [
          "data_tmp"
        ],
        "method_config_file": "null"
      },
      {
        "method_type": "PassThrough",
        "unique_name": "pass_through_2",
        "inputs": [
          "data_tmp"
        ],
        "outputs": [
          "output"
        ],
        "method_config_file": "null"
      }
    ]
}
```

### 定义PassThrough Method
现在需要定义PassThrough Method并实现其核心功能。
```c++
std::vector<std::vector<BaseDataPtr>> PassThrough::DoProcess(
    const std::vector<std::vector<BaseDataPtr>> &input,
    const std::vector<xstream::InputParamPtr> &param) override {
  return input;
}
```

### 帧率统计
程序中通过异步接口`AsyncPredict()`将输入数据送入框架进行运算，通过回调函数进行帧率的统计。

我们设置了大小为50的"缓冲窗口"，当任务数量大于50时，会等待任务数量完成至50内再继续输入数据。
```c++
  xstream::XStreamSDK *flow = xstream::XStreamSDK::CreateSDK();
  flow->SetConfig("config_file", config);
  flow->SetCallback(std::bind(&FrameFPS, std::placeholders::_1));
  flow->Init();

  while (!exit_) {
    InputDataPtr inputdata(new InputData());
    BaseDataPtr data(new BaseData());
    data->name_ = "input";   // corresponding the inputs in workflow
    inputdata->datas_.push_back(data);

    {
      std::unique_lock<std::mutex> lck(mtx);
      cv.wait(lck, [] {return task_num < 50;});
    }

    // async mode
    flow->AsyncPredict(inputdata);
    task_num++;
  }
```

帧率统计函数定义如下：
```c++
void FrameFPS(xstream::OutputDataPtr output) {
  static auto last_time = std::chrono::system_clock::now();
  static int fps = 0;
  static int frameCount = 0;

  frameCount++;

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now() - last_time);
  if (duration.count() > 1000) {
    fps = frameCount;
    frameCount = 0;
    last_time = std::chrono::system_clock::now();
    std::cout << "fps = " << fps << std::endl;
  }
  --task_num;
  std::unique_lock <std::mutex> lck(mtx);
  cv.notify_one();
}
```

运行benchmark_main程序后，会持续输出帧率大小，用户可以通过`ctrl+C`来终止。
```
PassThrough Init
PassThrough Init
fps = 5180
fps = 5171
fps = 5165
fps = 5185
fps = 5190
fps = 5160
fps = 5157
fps = 5169
fps = 5173
fps = 5159
fps = 5175
fps = 5174
^Crecv signal 2, stop
PassThrough Finalize
PassThrough Finalize
```
