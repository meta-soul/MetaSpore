# metasporeflow-online
##一，实现功能
> metasporeflow 在线服务部分主要实现功能：
>>1. 根据配置生成在线服务任务的docker-compose文件并启动
>>2. 根据配置生成推荐服务recommend-service的配置并推送到consul且通知recommend-service服务
##二，配置说明
> metasporeflow 在线服务配置包括
>+ docker配置：
>   - dockers：dockers是一个以服务名称为key，DockerInfo为value的字典
>>DockerInfo 主要配置在线服务任务的docker信息，目前支持配置服务的image和服务启动环境变量environment  
>>**其中服务名称包括**
>>* recommend-service名称为recommend
>>* consul服务名称为consul
>>* model-serving服务名称要求前缀为model的字符串 （支持部署多个model serving）  
>
>+ 外部服务配置：
>   - services: 是一个以服务名称为key，服务访问信息DataSource为value的字典  
>>DataSource配置服务依赖的服务访问地址，必须使用kind指定外部服务的类型，目前支持MongoDB，Redis，JDBC。 
>>demo中使用mongodb，配置如下：

     mongo:
       host: 172.17.0.1
       port: 27018
       kind: mongodb
       collection: [jpa]
       options:
         uri: mongodb://user:password@172.17.0.1:27018/jpa?authSource=admin
        
>+ 特征数据配置：
>   - source： 包括user数据， item数据， summary数据以及相关字段信息  
>>每一份数据需要配置访问数据表，依赖数据服务名称，访问数据库， 表结构  
>>demo中user， item， item_summary相关配置如下：

    user:
       table: amazonfashion_user_feature
       serviceName: mongo
       collection: jpa
       columns:
         - user_id: str
         - user_bhv_item_seq: str
     item:
       table: amazonfashion_item_feature
       serviceName: mongo
       collection: jpa
       columns:
         - item_id: str
         - brand: str
         - category: str
     summary:
       table: amazonfashion_item_summary
       serviceName: mongo
       collection: jpa
       columns:
         - item_id: str
         - brand: str
         - category: str
         - title: str
         - description: str
         - image: str
         - url: str
         - price: str
>>需要配置request请求数据的数据结构，包括字段和类型, 如
        
    request:
       - user_id: str
       - item_id: str
       
>>同时配置user的主键字段名称如user_id, item的主键字段名称如item_id, user相关的item列表字段名称， item列表数据默认分隔符
>+ 模型数据配置：
>   - cf_models：cf模型包括itemcf，swing等
>>配置模型名称和数据源信息，其中数据源信息包括数据表名称， 数据服务名称， 数据库， 表结构等，表结构一般包括key， value两个字段  
>>demo中配置swing，其中表结构使用默认结果key:str, value: {list_struct: { item_id: str, score: double}}

    name: swing
       source:
         table: amazonfashion_swing
         serviceName: mongo
         collection: jpa
>   - random_models: 随机模型一般用于抄底召回
>>随机模型相比于cf模型，需要配置bound，指定随机数范围，比如pop抄底召回随机范围为10
>   - twotower_models:双塔模型
>>双塔模型需要配置
>>* 双塔模型名称
>>* 用于生成user embedding的模型名称
>>* milvus地址信息
>   - rank_models:排序模型
>>排序模型需要配置
>>* 名称
>>* 所使用的模型名称
>>* 模型输入字段信息
>>* 交叉特征列表
>>demo中配置widedeep相关配置如下：

       name: widedeep
       model: amazonfashion_widedeep
       column_info:
         - dnn_sparse: ["user_id", "item_id", "brand", "category"]
         - lr_sparse: ["user_id", "item_id", "category", "brand", "user_id#brand", "user_id#category"]
       cross_features:
         - name: user_id#brand
           join: "#"
           fields: ["user_id", "brand"]
         - name: user_id#category
           join: "#"
           fields: [ "user_id", "category" ]
##三，执行过程说明
>1. 作为metasporeflow的一部分，服务的up执行需要通过metaspore flow up来启动  
>* 启动过程中，会输出一些配置相关的debug信息  
>* 程序先启动docker相关服务，会输出启动docker服务的任务返回信息CompletedProcess， returncode=0说明docker服务启动成功，其他则表示失败，相关信息见stderr
>* 然后生成recommend service的配置， 通过循环重试提交配置到consul上，成功提交配置会输出 set config to consul success!
>* 最后通知recommend service更新配置，通过循环重试请求recommend service的refresh接口，成功刷新会输出一个list [], list中的内容表示通过refresh使得在线服务更新的配置内容
>* 在线部分执行完毕会输出online local flow up
>2. 执行成功后，可以通过curl命令验证服务正确性，可以通过docker logs container_recommend_service查看在线服务日志