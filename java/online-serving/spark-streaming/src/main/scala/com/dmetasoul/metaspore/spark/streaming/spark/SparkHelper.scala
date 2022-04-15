package com.dmetasoul.metaspore.spark.streaming.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object SparkHelper {
  def getAndConfigureSparkSession() = {
    val conf = new SparkConf()
      .setAppName("Structured Streaming from Kafka to Local File")
      .setMaster("local[2]")
      .set("spark.sql.streaming.checkpointLocation", "checkpoint")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    SparkSession
      .builder()
      .getOrCreate()
  }

  def getSparkSession() = {
    SparkSession
      .builder()
      .getOrCreate()
  }
}
