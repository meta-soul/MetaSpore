# Tracking

提供一个 Http 接口的 Web 服务，用来将实验的请求日志，异步的发送至 Kafka。

<br/>

- 提供 Http 接口接收实验日志
- 异步发送日志到 Kafka
- 自动创建 Default Kafka Topic 

<br/>

## application.yml 配置

- 配置 **default-kafka-topic**，可以自动创建 Kafka Topic

- 配置 Kafka 集群信息，其中 spring.profiles.active 为启动后访问的集群

<br/>

```yaml
default-kafka-topic:
  name: "default-kafka-topic"
  partition: 2
  replication: 2
---
spring:
  profiles:
    active: local
---
spring:
  config:
    activate:
      on-profile: local
  kafka:
    template:
      default-topic: "default-kafka-topic"
    producer:
      bootstrap-servers: localhost:9092
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer
      properties:
        acks: all
        retries: 10
        retry.backoff.ms: 1000
    admin:
      properties:
        bootstrap.servers: localhost:9092
---
spring:
  config:
    activate:
      on-profile: nonprod
  kafka:
    template:
      default-topic: test
    producer:
      bootstrap-servers: localhost:9095,localhost:9096,localhost:9097
      key-serializer: org.apache.kafka.common.serialization.IntegerSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer
      ssl:
        trust-store-location: YOUR-TRUSTSTORE-PATH/client.truststore.jks
        trust-store-password: password
        key-store-location: YOUR-KEYSTORE-PATH/client.keystore.jks
        key-store-password: password
      properties:
        acks: all
        retries: 10
        retry.backoff.ms: 1000
        security:
          protocol: SSL
        ssl.endpoint.identification.algorithm:

---
spring:
  config:
    activate:
      on-profile: prod
  kafka:
    producer:
      bootstrap-servers: prod:9092
      key-serializer: org.apache.kafka.common.serialization.IntegerSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer

```

<br/>

## Http 示例

<br/>

1. 启动 tracking 服务
2. 实验端发送 Http 请求至 Tracking 服务。

```
curl -i \
-d '{"requestId":"aaa","topic":"default-kafka-topic","customData":{"requestId":"aaa"}}' \
-H "Content-Type: application/json" \
-X POST http://localhost:8080/v1/tracking
```

3. 消费 Kafka 的 Topic 数据进行测试

```
kafka-console-consumer.sh --bootstrap-server localhost:9092 --from-beginning --topic default-kafka-topic
```
<br/>