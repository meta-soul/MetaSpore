多模态 Demo 线上服务部分由以下几部分构成：

- `multimodal_serving`，多模态示例的算法服务，含有实验配置、预处理、召回、排序等整条算法处理链路
- `multimodal_preprocess`，对多模态大模型预处理逻辑（含文本/图像等）的封装，以 gRPC 接口提供服务
