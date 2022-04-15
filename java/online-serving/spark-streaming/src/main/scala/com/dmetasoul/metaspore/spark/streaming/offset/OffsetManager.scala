package com.dmetasoul.metaspore.spark.streaming.offset

import org.apache.spark.streaming.kafka010.OffsetRange

/**
 *
 * @author: Hades-888
 * @date: 2022/4/13 14:37
 * @description:
 *
 */

// TODO: 实现 exectly once consumer, 实现从 DB 中访问偏移量
trait OffsetManager {

  def readOffsets(): Unit

  def saveOffsets(offsetRanges: Array[OffsetRange]): Unit
}
