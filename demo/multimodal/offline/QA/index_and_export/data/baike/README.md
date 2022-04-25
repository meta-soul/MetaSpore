Download from: https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa

`baike_qa_1w.json` 是随机抽取的 1w 测试数据，可以先尝试该数据来跑通离线建库流程。

百科问题数据描述：

含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别，其中频率达到或超过10次的类别有434个。

数据集划分：数据去重并分成三个部分。训练集：142.5万；验证集：4.5万；测试集，数万，不提供下载。

可能的用途：
可以做为通用中文语料，训练词向量或做为预训练的语料；也可以用于构建百科类问答；其中类别信息比较有用，可以用于做监督训练，从而构建

更好句子表示的模型、句子相似性任务等。
结构：
{"qid":<qid>,"category":<category>,"title":<title>,"desc":<desc>,"answer":<answer>}

其中，category是问题的类型，title是问题的标题，desc是问题的描述，可以为空或与标题内容一致。
