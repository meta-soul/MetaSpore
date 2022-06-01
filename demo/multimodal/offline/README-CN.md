这里主要是 demo 示例对应的**离线代码**，含离线模型训练、导出、建库等：

- [QA](./QA) 是以[百科问答数据](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa)为基础的**以文搜文**演示样例
- [txt2img](./txt2img) 是以 [Unsplash Lite](https://unsplash.com/data) 图片库数据为基础的**以文搜图**演示样例

------

此外我们还针对 HuggingFace NLP/CV 预训练模型提供了一套通用的模型上线规范，只要按照规范把模型导出就可以在 MetaSpore Serving 上进行模型推理，具体规范以及模型导出演示样例参考[此处](./model_export)。
