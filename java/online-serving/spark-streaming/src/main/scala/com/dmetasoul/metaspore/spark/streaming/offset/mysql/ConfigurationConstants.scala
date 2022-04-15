package com.dmetasoul.metaspore.spark.streaming.offset.mysql

import java.text.SimpleDateFormat

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Duration, Milliseconds}
import java.util.Date

import org.apache.spark.sql.types.{DataTypes, StructField, StructType}

/**
  * Created by Administrator on 2017/4/14.
  */
object ConfigurationConstants {
  val sparkSinkHost = "xxx.xxx.xxx.xx"
  val sparkSinkPort = 122345
  val streamingInterval = Milliseconds(10000)
  val checkpointInterval = Duration(30000)
  val streamingAppName = "myapp"
  val deployMode = "yarn-client"
  val checkpointDirectory = "hdfs://hadoopcluster/tmp/sparkstreaming"
  val streamingStorageLevel = StorageLevel.MEMORY_AND_DISK_SER_2
  val kafkaTopic = "realtime1"
  val kafkaConsumerTopics = "streaming_test"
  val kafkaBrokers = "xxx.xxx.xxx.xx:6667,xxx.xxx.xxx.xx:6667,xxx.xxx.xxx.xx:6667"
  val kafkaClientID = "ScalaProducerExample"
  val kafkaKeySer = "org.apache.kafka.common.serialization.StringSerializer"
  val kafkaValueSer = "org.apache.kafka.common.serialization.StringSerializer"
  val clearInterval = 86400000L
  val clearDate: Date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse("2017-04-20 00:00:00")
  val middleTableName1 = "RealTimeAd"
  val middleTableSchema1 = StructType(
    Array(StructField("ad_id", DataTypes.StringType, false),
      StructField("pv", DataTypes.IntegerType, true),
      StructField("uv", DataTypes.IntegerType, true),
      StructField("current_time", DataTypes.StringType, true))
  )
  val middleTableName2 = "RealTimeGame"
  val middleTableSchema2 = StructType(
    Array(StructField("channel_id", DataTypes.StringType, false),
      StructField("game_count", DataTypes.IntegerType, true),
      StructField("user_count", DataTypes.IntegerType, true),
      StructField("current_time", DataTypes.StringType, true))
  )
  val middleTableName = "RealTime"
  val middleTableSchema = StructType(
    Array(StructField("channel_id", DataTypes.StringType, false),
      StructField("pv", DataTypes.IntegerType, true),
      StructField("uv", DataTypes.IntegerType, true),
      StructField("game_count", DataTypes.IntegerType, true),
      StructField("user_count", DataTypes.IntegerType, true),
      StructField("current_time", DataTypes.StringType, true))
  )

  val connectOptions = Map(
    "url" -> "jdbc:mysql://192.168.110.232:3306/tab_ad_result_develop?useUnicode=true&characterEncoding=utf-8",
    "driver" -> "com.mysql.jdbc.Driver",
    "user" -> "oss",
    "password" -> "vUBerOnm",
    "dbtable" -> "ad_h5game_base_info"
  )
}
