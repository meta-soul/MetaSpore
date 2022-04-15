package com.dmetasoul.metaspore.spark.streaming.offset

/**
 *
 * @author: Hades-888
 * @date: 2022/4/13 15:50
 * @description:
 *
 */
case class OffsetSchema(topic: String, group: String, step: Int, partition: Int, from: Long, until: Long, count: Long, datetime: String)
