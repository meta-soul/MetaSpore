# recommend-service 配置
## 一. 简介
> recommend-service 配置主要包括两部分特征生成部分和推荐服务部分

## 二. 特征生成配置：feature-config
> feature-config包括 
> + 特征数据连接信息配置：source
> + 特征数据表配置：sourceTable
> + 特征join关联表配置： feature
> + 特征计算转换配置： algoTransform

### 1. 特征数据连接信息配置：source
> source包含配置项：
> + name：str, 必须字段，数据连接名称
> + kind：str, 可选字段(默认为Request)，暂时支持Request， JDBC， MongoDB， Redis （区分大小写）
> + options: map<str, object>, 可选字段(只有Request条件下可不配置)  
>>- 对于JDBC 至少必须配置uri, driver; 
>>- 对于Redis 必须配置standalone，sentinel，cluster中一项， 每一项至少包括host， port等配置；
>>- 对于MongoDB 至少配置uri  

> source支持配置多个同kind的数据源，后续根据需要kind可以增加对es等更多数据源的支持。

### 2. 特征数据表配置：sourceTable
> sourceTable包含配置项：
> + name：str, 必须字段，特征数据表数据别名
> + source：str, 必须字段，特征数据连接信息source中配置的name
> + columns：list<map<str, object>>, 必须字段， 特征数据表的表结构， 列表中每一项为一个key-value对，key是字段名， value是类型名称，支持struct类型配置，如
```
    value_list: # 字段名
        list_struct: # 表示是一个list， 元素为一个struct，这个struct包含一个str字段item_id, double字段score
          item_id: str
          score: double
```
> + options: map<str, object>,可选字段（默认为空字典), 读取数据表额外需要的配置参数
> + taskName：str, 可选字段（默认跟随kind选择相应的读取程序），一般不用配置，用于自定义数据源读取程序时使用
> + table： str, 可选字段（默认为name)， 数据表名称 (没有数据表 如redis，可不配置)
> + prefix： str, 可选字段（默认为"")，暂时特别用于redis， 生成redis key时，指定key的格式， 字符串format格式化字符串，可以不用配置
> + sqlFilters: list<str>, 可选字段（默认为[]), 特别用于JDBC条件下， 列表中每一项为sql中where过滤条件sql片断
> + filters: list<map<str, map<str, object>>>，可选字段（默认为[]), 特别用于MongoDB条件下， 列表中每一项为一个key-value对， key是数据表包含的字段名称， value是一个map， 其中key支持gt, lt, ge, le, eq, in等操作， value是相应的数值

### 3. 特征join关联表配置： feature
> feature包含配置项：
> + name：str, 必须字段，特征join关联表名称
> + from：list<str>, 必须字段，参与关联的数据表，数据表可以是sourceTable，algoTransform中配置的表name，也可以服务流程中的带schema信息的中间数据表，如rank中recall数据
> + select: list<str>, 必须字段，关联表选取的字段，对于from表里有多个表同时含有相同字段名称时，需要指定table
> + condition: list<struct>, 可选字段（默认为[]), list中每一项包含一个key-value，key是join左表表名.字段， value是join右表表名。字段； type指定join类型，如left， right， inner， full。 默认inner
> + filters:list<map<str, map<str, object>>>,可选字段（默认为[]), 类似于sourceTable中的filters，不同的是这里对所有kind都生效，暂不支持类似a.field1 > b.fileld2这样的过滤

### 4. 特征计算转换配置： algoTransform
> algoTransform包含配置项：
> + name：str, 必须字段，数据连接名称
> + taskName：str, 可选字段（默认为AlgoTransform)，根据特征业务需求选择。可自定义扩展，暂时系统支持：
>>- ItemMatcher （用于处理召回数据）可通过options中maxReservation控制召回数量， options中algo-name填写召回算法名称
>>- UserProfile （用于处理user feature，暂时只针对当前推荐demo）可通过options中alpha控制用户关联item id的权重， options中splitor用于指定用户关联item id字符串分隔符
>>- AlgoInference （用于请求model serving）可通过options中modelName指定模型名称；options中targetKey填写模型输出key，默认output；options中targetIndex默认-1， host：指定model serving地址， port：指定model serving port，默认50000等
>>- MilvusSearch （用于处理milvus请求，双塔召回）可通过options中maxReservation控制milvus请求数量， options中collectionName填写milvus数据库名称；host：指定milvus地址， port：指定milvus port，默认19530等
> + feature: list<str>, 可选字段（默认为[])，依赖的关联表数据,
> + algoTransform: list<str>, 可选字段（默认为[])，依赖的计算特征表
> + fieldActions:必须字段，指定字段处理逻辑, 其中FieldAction 输出由names指定， 输入为fields+input（fields在前）。其配置如下：
>> - names: list<str>, 必须字段，FieldAction的输出字段名
>> - types: list<object>, 必须字段，FieldAction的输出字段的类型（类型支持如sourceTable中columns的类型一样）
>> - fields: list<str>, 可选字段（默认为[]),来自于依赖feature和algoTransform表里的字段名
>> - input: list<str>, 可选字段（默认为[]),来自于本algoTransform的其他FieldAction的输出字段名（中间计算结果, input中字段不能与names中字段重合)
>> - func: str, 可选字段（默认为""),处理函数名称,
>> - algoColumns: map<str, list<str>>, 可选字段（默认为{}),特用于AlgoInference，指定模型输入字段信息，如：
```
      algoColumns:
      - dnn_sparse:
        - user_id
        - item_id
        - brand
        - category
      - lr_sparse:
        - user_id
        - item_id
        - category
        - brand
        - user_id#brand
        - user_id#category
```
> + output:list<str>, 必须字段，algoTransform最终的输出字段
> + options: map<str, object>, 可选字段（默认为空字典),相关额外参数

## 三. 推荐服务：recommend-config
> recommend-config包括 
> + 推荐服务模块：services
> + 推荐实验：experiments
> + 推荐实验层： layers
> + 推荐场景： scenes

### 1. 推荐服务模块：services
> services包含配置项：
> + name：str, 必须字段，推荐服务模块名称
> + taskName：str, 可选字段（默认为系统实现)，用于自定义服务实现
> + tasks: list<str>, 可选字段（默认[]), 服务模块依赖的特征计算任务，比如swing召回，依赖algotransform_swing
> + preTransforms: list<struct>， 可选字段（默认[]), 定义在服务处理之前对传递给本服务的数据进行预处理的操作，比如多份数据汇总
> + transforms: list<struct>， 可选字段（默认[]), 定义在服务处理之后对传递给下游服务的数据进行处理的操作，比如截断，排序， 过滤
> + options: map<str, object>,可选字段（默认为空字典),定义额外的参数

### 2. 推荐实验：experiments
> experiments包含配置项：
> + name：str, 必须字段，推荐实验名称
> + taskName：str, 可选字段（默认为系统实现)，用于自定义服务实现
> + chains: list<chain> 必须字段, chain中配置：
>> - name: str， 任务chain名称，可选参数
>> - then: list<str>,可选参数，顺序执行任务列表，每一项任务为services中定义的推荐服务名称
>> - when: list<str>,可选参数，并发执行任务列表，每一项任务为services中定义的推荐服务名称
>> - isAny: boolean, 可选参数，默认False， 指定when并发任务运行是否等待任务全部完成， False表示需要等到全部任务完成。
>> - timeOut: long, 可选字段， 默认30000， 指定并发任务完成超时时间，单位毫秒
>> - transforms: list<struct>， 可选字段（默认[]), 定义在服务处理之后对传递给下游服务的数据进行处理的操作，比如截断，排序， 过滤
> + options:map<str, object>,可选字段（默认为空字典),定义额外的参数

### 3. 推荐实验层： layers
> layers包含配置项：
> + name：str, 必须字段，推荐实验层名称
> + taskName：str, 可选字段（默认为系统实现)，用于自定义服务实现
> + experiments: list<struct>, 定义实验列表，列表每一项包含实验占比,
> + bucketizer: str， 必须字段， 默认为random，指定划分函数
> + options:map<str, object>,可选字段（默认为空字典),定义额外的参数

### 4. 推荐场景： scenes
> scenes包含配置项：
> + name：str, 必须字段，推荐场景名称
> + taskName：str, 可选字段（默认为系统实现)，用于自定义服务实现
> + chains: list<chain> 必须字段, 同推荐实验experiments中的chains， 不过chain中的任务为实验层layer
> + options:map<str, object>,可选字段（默认为空字典),定义额外的参数

## 三. demo环境下服务配置
> 在ecommerce demo环境下，online服务配置基于一些固定的流程，因此配置有所简化。具体需要配置的内容如下：  
> ### 1. feature生成部分：
> + source：数据源信息， 例如：
```
     mongo:
       host: 172.17.0.1
       port: 27018
       kind: MongoDB
       collection: [jpa]
       options:
         uri: mongodb://user:password@172.17.0.1:27018/jpa?authSource=admin
```
> + sourceTable: 作为一个商品系统，需要有用户特征表，商品特征表， 商品详情表, 以及request数据结构。 例如：
```
    request:
      - user_id: str
      - item_id: str
    user:
       table: amazonfashion_user_feature
       source: mongo
       columns:
         - user_id: str
         - user_bhv_item_seq: str
     item:
       table: amazonfashion_item_feature
       source: mongo
       columns:
         - item_id: str
         - brand: str
         - category: str
     summary:
       table: amazonfashion_item_summary
       source: mongo
       columns:
         - item_id: str
         - brand: str
         - category: str
         - title: str
         - description: str
         - image: str
         - url: str
         - price: str
```
> + feature: 需要配置用户id的字段名称，使得用户特征表能与request数据关联； 需要商品id的字段名称用于商品数据表的join关联。关联字段的数据类型需要匹配。
```
    user_key_name: user_id
    item_key_name: item_id
```
> feature_user， feature_item_summary  
>> 由于demo环境下，特征惯量表流程已经确定， 当指定user_key_name后，即可确定user的关联表：feature_user用于生成用户特征; 当指定item_key_name后，即可确定item的关联表：feature_item_summary用于item详情
> + algotransform: 需要配置UserProfile任务中处理用户行为关联item列表的字段和itemId分隔符；需要请求model serving需要的模型名和模型输入特征， 以及配置交叉特征列表用于预先生成交叉特征数据
```
    user_item_ids_name: user_bhv_item_seq
    user_item_ids_split: "\u0001"
```
> feature_user->algotransform_user
>> 当指定user_item_ids_name和user_item_ids_split后，即可确定user的特征计算表：algotransform_user. 根据feature_user中的user_bhv_item_seq取出字段使用"\u0001"分隔字符串解析出item id列表，然后计算出相应的item权重
> ### 2. 配置模型相关feature
> + source：这里使用上面的mongodb
> + 补召回模型
```
  random_models:
    - name: pop
      bound: 10
      source:
        table: amazonfashion_pop
        source: mongo
        columns:
          - key: int
          - value_list:
              list_struct:
                item_id: str
                score: double
```
>algotransform_random->feature_pop->algotransform_pop
>> 根据bound: 10配置， 生成包含user id和随机生成数的algotransform_random， 基于algotransform_random和amazonfashion_pop关联生成包含user_id和pop itemIds的feature_pop, 再基于feature_pop生成pop的召回数据algotransform_pop
> + cf召回模型
```
  cf_models:
    - name: swing
      source:
        table: amazonfashion_swing
        source: mongo
        columns:
          - key: int
          - value:
              list_struct:
                _1: str
                _2: double
```
> algotransform_user->itemIds的feature_swing -> algotransform_swing
>> 基于algotransform_user和amazonfashion_swing关联生成包含user_id和swing itemIds的feature_swing, 再基于feature_swing生成swing的召回数据algotransform_swing  

> + 排序模型
```
  rank_models:
    - name: widedeep
      model: amazonfashion_widedeep
      algoColumns:
        - dnn_sparse: ["user_id", "item_id", "brand", "category"]
        - lr_sparse: ["user_id", "item_id", "category", "brand", "user_id#brand", "user_id#category"]
      cross_features:
        - name: user_id#brand
          join: "#"
          fields: ["user_id", "brand"]
        - name: user_id#category
          join: "#"
          fields: [ "user_id", "category" ]
```
> feature_widedeep->algotransform_widedeep
>> 基于召回数据与amazonfashion_item_feature关键生成feature_widedeep, 基于feature_widedeep先生成交叉特征，然后基于feature_widedeep和交叉特征数据请求model serving生成widedeep模型排序结果数据：algotransform_widedeep
> ### 3. 推荐流程部分（目前demo环境下，全部为默认配置，每个模型一路实验，分recall和rank两个实验层，每一层里平均划分流量给实验)：
>> + services: 配置实验名称，配置依赖服务，demo环境里为algotransform_模型名， rank服务，需要接收上游recall传递过来的召回数据，需要配置召回数据的格式，方便特征计算关联表
```
  - name: recall_swing:
    tasks: 
    - algotransform_swing
    options:
      maxReservation: 200
  - name: rank_widedeep
    preTransforms:
    - name: summary
    columns:
    - user_id: str
    - item_id: str
    - score: double
    - origin_scores: map_str_double
    tasks:
    - algotransform_widedeep
    options:
      maxReservation: 200
```
> + experiments：每个实验的召回或排序等服务组合
```
  - name: recall.swing
    options:
      maxReservation: 100
    chains:
    - then: recall_swing
  - name: rank.widedeep
    options:
      maxReservation: 100
    chains:
    - then: rank_widedeep
```
> + layers: 实验以及占比
```
  - name: recall
    bucketizer: random
    experiments:
    - name: recall.swing
      ratio: 1.0
  - name: rank
    bucketizer: random
    experiments:
    - name: rank.widedeep
      ratio: 1.0
```
> + scenes: 配置场景
```
  - name: guess-you-like
    chains:
    - then:
      - recall
      - rank
  - name: looked-and-looked
    chains:
    - then:
      - related_recall
      - rank
```