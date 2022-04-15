package com.dmetasoul.metaspore.spark.streaming.kafka

import com.dmetasoul.metaspore.spark.streaming.spark.SparkHelper
import org.apache.spark.sql.types.{LongType, StringType, StructType}

/**
 *
 * @author: Hades-888
 * @date: 2022/4/13 11:49
 * @description:
 *
 */
object KafkaService {
  private val spark = SparkHelper.getSparkSession()

  val radioStructureName = "consume-kafka-test"

  val topicName = "default-kafka-topic"

  val bootstrapServers = "localhost:9092"

  val schemaOutput = new StructType()
    .add("title", StringType)
    .add("artist", StringType)
    .add("radio", StringType)
    .add("count", LongType)
}
