package com.dmetasoul.metaspore.spark.streaming.offset.mysql

import com.dmetasoul.metaspore.spark.streaming.offset.OffsetManager
import org.apache.kafka.clients.consumer.ConsumerConfig
import org.apache.kafka.common.TopicPartition
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.kafka010._

import java.sql.PreparedStatement


/**
 *
 * @author: Hades-888
 * @date: 2022/4/13 18:07
 * @description:
 *
 */
class MysqlOffsetManagerImpl extends OffsetManager {

  override def readOffsets(): Unit = {}

  override def saveOffsets(offsetRanges: Array[OffsetRange]): Unit = {

    val connOffset = ConnectPool.getConnection
    connOffset.setAutoCommit(false)
    val psOffset: PreparedStatement = connOffset.prepareStatement("REPLACE INTO `kafka_offset` (`topic`, `partition`, `offset`) VALUES (?,?,?)")
    for (o <- offsetRanges) {
      println(s"${o.topic} ${o.partition} ${o.fromOffset} ${o.untilOffset}")
      psOffset.setString(1, o.topic.toString)
      psOffset.setInt(2, o.partition.toInt)
      psOffset.setLong(3, o.fromOffset.toLong)
      psOffset.addBatch()
    }
    psOffset.executeBatch()
    connOffset.commit()
    ConnectPool.closeCon(psOffset, connOffset)
  }


  def kafkaOffsetRead(ssc: StreamingContext, kafkaParams: Map[String, String], consumerTopics: Set[String]): InputDStream[(String, String)] = {
    val connOffset = ConnectPool.getConnection
    val psOffsetCnt: PreparedStatement = connOffset.prepareStatement("SELECT SUM(1) FROM `kafka_offset` WHERE `topic`=?")
    psOffsetCnt.setString(1, ConfigurationConstants.kafkaConsumerTopics)
    val rs = psOffsetCnt.executeQuery()
    var parCount = 0
    while (rs.next()) {
      parCount = rs.getInt(1)
      println(parCount.toString)
    }

    val brokers = ""
    val groupId = ""
    val topics = ""

    // Create direct kafka stream with brokers and topics
    val topicsSet = topics.split(",").toSet
    val kafkaParams = Map[String, Object](
      ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG -> brokers,
      ConsumerConfig.GROUP_ID_CONFIG -> groupId,
      ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG -> classOf[StringDeserializer],
      ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG -> classOf[StringDeserializer])


    // begin from the offsets committed to the database
    //    val fromOffsets = selectOffsetsFromYourDatabase.map {
    //      resultSet =>
    //        new TopicPartition(resultSet.string("topic"), resultSet.int("partition")) -> resultSet.long("offset")
    //    }.toMap

    //    val stream = KafkaUtils.createDirectStream[String, String](
    //      ssc,
    //      LocationStrategies.PreferConsistent,
    //      Assign[String, String](fromOffsets.keys.toList, kafkaParams, fromOffsets)
    //    )
    return null
  }

}

