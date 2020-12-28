## 检测 + 分类 模型集成示例
本节内容提供了集成检测和分类两种模型的示例，其中检测模型使用yolov3，分类模型使用mobilenetv2。注意，本示例仅为集成模型提供参考，对模型的精度等不做保证。

模型集成的关键是Method的开发，这里我们xstream-framework框架内已经提供了集成智能模型的基础类`DnnPredictMethod`和`DnnPostProcessMethod`。

其中DnnPredictMethod中包括虚函数`PrepareInputData()`，核心操作是对输入数据进行预处理、复制到BPU输入input_tensors，并添加预测任务，需要根据模型实现该函数内容。
```c++
  int PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      std::vector<std::vector<BPU_TENSOR_S>> &input_tensors,
      std::vector<std::vector<BPU_TENSOR_S>> &output_tensors) {
    return -1;
  }

  virtual int PrepareInputData(
      const std::vector<BaseDataPtr> &input,
      const std::vector<InputParamPtr> param,
      hobot::vision::PymImageFrame &pyramid,
      std::vector<BPU_BBOX> &input_bbox,
      std::vector<int> &valid_box,
      std::vector<BPU_TENSOR_S> &output_tensors) {
    return -1;
  }
```

DnnPostProcessMethod中包括虚函数`ParseDnnResult()`，主要操作是根据BPU输出数据解析结果，并封装成框架支持的BaseData数据类型。
```c++
  int ParseDnnResult(DnnAsyncData &dnn_result,
                     std::vector<BaseDataPtr> &frame_result) {
    return -1;
  }
```

### 模型信息描述
YoloV3模型文件yolov3_nv12_hybrid_horizonrt.bin，模型信息描述如下【可以通过bpu_predict接口获取模型信息】：
```
Input num:1：
input[0]: data type: BPU_TYPE_IMG_YUV_NV12, shape:(1,3,416,416), layout: BPU_LAYOUT_NCHW, aligned shape:(1,4,416,416)

Output num:3
output[0]: data type: BPU_TYPE_TENSOR_F32, shape:(1,13,13,255), layout: BPU_LAYOUT_NHWC, aligned shape:(1,13,13,255)
output[1]: data type: BPU_TYPE_TENSOR_F32, shape:(1,26,26,255), layout: BPU_LAYOUT_NHWC, aligned shape:(1,26,26,255)
output[2]: data type: BPU_TYPE_TENSOR_F32, shape:(1,52,52,255), layout: BPU_LAYOUT_NHWC, aligned shape:(1,52,52,255)
```

MobilenetV2模型文件mobilenetv2_nv12_hybrid_horizonrt.bin，模型信息描述如下：
```
Input num:1：
input[0]: data type: BPU_TYPE_IMG_YUV_NV12, shape:(1,3,224,224), layout: BPU_LAYOUT_NCHW, aligned shape:(1,4,224,224)

Output num:1
output[0]: data type: BPU_TYPE_TENSOR_F32, shape:(1,1000,1,1), layout: BPU_LAYOUT_NCHW, aligned shape:(1,1000,1,1)
```

### 金字塔概念说明
金字塔包含基本层与缩放层：
 - 基本层为0，4，8，12，16，20这6层，图像分辨率分别为原图、原图/2、原图/4等等，基本层缩放比例固定。 
 - 其他层为缩放层，缩放层的缩放范围可以配置：金字塔1-3层，为基于基本层0的缩放层；金字塔5-7层，为基于基本层4的缩放层；以此类推。通过设置`roi_x_i、roi_y_i、roi_w_i、roi_h_i`，从对应基本层中扣取该区域ROI；`factor_i`用于表示相对对应的基本层，需要缩放的倍数。缩放倍数计算公式为： 64/(64+factor)， factor取值范围为[1-63], factor设置为0表示该层不使能。最终该缩放层的大小为： ROI * 64/(64+factor)，宽与高分别向下取整。

以原图1920x1080为例，金字塔第0层大小为1920x1080，第4层为960x540，第8层为480x270，以此类推。

### 集成yolov3检测模型
根据上述说明，集成一个模型需要添加预测和后处理两个Method，以模型YoloV3为例，我们需要集成Yolov3PredictMethod、Yolov3PostProcessMethod。根据模型信息，yolov3输入是416x416大小的nv12数据，因此可以直接使用金字塔数据。在Yolov3PredictMethod中，需要在配置文件中**配置金字塔层数**，并在初始化中读取配置信息。

#### 金字塔配置
检测模型输入的大小是416x416。图像原图为1920x1080，或者3840x2160。

若对原图直接缩放，缩放至416x416，势必会产生形变。为了保证模型效果，我们对原图分析区域做了裁剪与缩放，思路为：只对原图中间正方形区域做预测。对原图做裁剪与缩放，基于金字塔来完成。

以原图1920x1080为例，金字塔第0层大小为1920x1080，第4层为960x540，第8层为480x270。yolov3模型输入大小是416x416，第8层太小，因此需要以第4层(960x540)为基础层，从第4层的中间抠取520x520，并缩放到416x416。金字塔5-7层的基础层都是第4层，这里我们按照这样的逻辑配置金字塔第7层为模型的输入数据。
```
"roi_x_7": 220,   // 960x540中roi左上角x坐标, 计算方式：(960 - 520) / 2
"roi_y_7": 10,    // 960x540中roi左上角y坐标, 计算方式：(540 - 520) / 2
"roi_w_7": 520,   // 960x540中roi宽
"roi_h_7": 520,   // 960x540中roi高
"factor_7": 16,   // 960x540中roi缩放系数, 64/(64+factor) = 64/(64+16) = 0.8, 520*0.8 = 416
```

以原图3840x2160为例，金字塔第0层大小为3840x2160，第4层为1920x1080，第8层为960x540，以此类推。
yolov3模型输入大小是416x416，因此需要以第8层(960x540)为基础层，从第8层的中间抠取520x520，并缩放到416x416。金字塔9-11层的基础层都是第8层，这里我们按照这样的逻辑配置金字塔第10层为模型的输入数据。
```
"roi_x_10": 220,   // 960x540中roi左上角x坐标, 计算方式：(960 - 520) / 2
"roi_y_10": 10,    // 960x540中roi左上角y坐标, 计算方式：(540 - 520) / 2
"roi_w_10": 520,   // 960x540中roi宽
"roi_h_10": 520,   // 960x540中roi高
"factor_10": 16,   // 960x540中roi缩放系数, 64/(64+factor) = 64/(64+16) = 0.8, 520*0.8 = 416
```

#### Yolov3PredictMethod
根据上节内容，配置对应的金字塔层，直接可以拿到416x416大小的数据，这里简单说明`PrepareInputData()`函数的逻辑：
```
// 参考common/xstream/framework/tutorials/stage10/method/Yolov3PredictMethod
1. 从指定金字塔层中取数据，并检查数据大小是否与模型输入大小(416x416)一致。若不一致需要报错，一般是vio金字塔配置错误;
2. 申请input_tensor和output_tensor, 基类已实现，可以直接调用;
3. 复制金字塔图像数据到input_tensor;
```

另外，由于模型输出的检测结果是基于输入数据的，所以后续需要缩放到原分辨率大小(后处理中)，这里预测前需要获取原图分辨率`GetSrcImageSize()`，实现该函数后，基类DnnPredictMethod中会将原图大小传递给后处理。

#### Yolov3PostProcessMethod
后处理中，核心工作是在函数`ParseDnnResult()`中解析bpu输出结果output_tensor。需要注意的是，这里直接解析出的检测结果是基于输入数据的分辨率，需要将其坐标映射回原图分辨率。以1080p为例，原图大小(金字塔第0层)1920x1080，金字塔第4层960x540，从金字塔第4层960x540抠取中间520x520大小，再缩放到416x416后输入模型预测。映射逻辑：
```
// 参考`common/xstream/framework/tutorials/stage10/method/Yolov3PostProcessMethod`
// 映射关系, 原图坐标(x', y'), 输出结果(x, y)
x' = [x * (520.0 / 416.0) + (960-520)/2] * (1980 / 960)
y' = [y * (520.0 / 416.0) + (540-520)/2] * (1080 / 540)
```

后处理解析的检测框使用`hobot::vision::BBox`数据结构来表示(定义在common/xstream/vision_type/include/horizon/vision_type/vision_type.hpp)。此外，在函数`ParseDnnResult()`中还需要将解析后的数据封装为xstream-frame框架支持的BaseData数据结构。由于全图可能存在多个目标，结果会有多个检测框，这里需要使用派生数据结构xstream::BaseDataVector，将检测框封装到成员变量datas_中，Yolov3PostProcessMethod的输出数据结构是xstream::BaseDataVector<xstream::XStreamData<hobot::vision::BBox>>.
```c++
struct BaseDataVector : public BaseData {
  BaseDataVector();
  std::vector<BaseDataPtr> datas_;
};

template<typename Dtype>
struct XStreamData : public BaseData {
  Dtype value;
  XStreamData() {}
  explicit XStreamData(const Dtype& val) {
    value = val;
  }
};
```

### 集成mobilenetv2分类模型
根据模型描述信息，该模型的输入是224x224大小的nv12数据，该分类模型需要将roi数据送入模型进行预测。因此在预测方法Mobilenetv2PredictMethod中，需要输入的数据包括金字塔数据以及Yolov3PostProcessMethod输出的roi数据。在函数`PrepareInputData()`中，核心操作是抠取原图上的roi数据，并缩放到224x224大小。
```
// PrepareInputData的逻辑
// 参考common/xstream/framework/tutorials/stage10/method/Mobilenetv2PredictMethod
1. 检查输入数据，包括金字塔图像image，以及roi框;
2. 获取金字塔原图数据，设置roi信息, 注意按照Yolov3PostProcessMethod输出的roi格式进行解析;
3. 申请input_tensor和output_tensor, 基类已实现，可以直接调用;
4. 调用bpu接口将原图中roi数据缩放到input_tensor中
```

后处理中，核心工作是在函数`ParseDnnResult()`中解析bpu输出结果output_tensor，并封装成框架兼容的BaseData数据类型。与Yolov3PostProcessMethod输出类似，这里同样用`hobot::vision::BBox`数据结构来表示分类结果。Mobilenetv2PostProcessMethod的输出数据结构是xstream::BaseDataVector<xstream::XStreamData<hobot::vision::BBox>>.
详细可参考`common/xstream/framework/tutorials/stage10/method/Mobilenetv2PostProcessMethod`.