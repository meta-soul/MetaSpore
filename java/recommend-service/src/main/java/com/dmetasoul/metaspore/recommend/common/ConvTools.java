package com.dmetasoul.metaspore.recommend.common;

import java.sql.Time;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.Date;

import static java.time.ZoneOffset.UTC;

public class ConvTools {
    public final static ZoneId zoneCN = ZoneId.of("Asia/Shanghai");
    public static Timestamp parseTimestamp(Object value) {
        if (value == null) return null;
        if (value instanceof Timestamp) {
            return (Timestamp) value;
        }
        if (value instanceof Long && (long)value >= 0) {
            return new Timestamp((long)value);
        }
        if (value instanceof Integer && (Integer)value >= 0) {
            return new Timestamp(((Integer) value).longValue() * 1000L);
        }
        if (value instanceof String) {
            try {
                return Timestamp.valueOf((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        if (value instanceof LocalDateTime) {
            return Timestamp.valueOf((LocalDateTime)value);
        }
        return null;
    }

    public static LocalDateTime timestamToDatetime(long timestamp){
        Instant instant = Instant.ofEpochMilli(timestamp);
        return LocalDateTime.ofInstant(instant, UTC);
    }

    public static LocalTime timestamToLocaltime(long timestamp){
        Instant instant = Instant.ofEpochMilli(timestamp);
        return LocalTime.ofInstant(instant, UTC);
    }

    public static LocalDateTime parseLocalDateTime(Object value) {
        if (value == null) return null;
        if (value instanceof LocalDateTime) {
            return (LocalDateTime) value;
        }
        if (value instanceof Long && (long)value >= 0) {
            return timestamToDatetime((Long) value);
        }
        if (value instanceof Integer && (Integer)value >= 0) {
            return timestamToDatetime(((Integer) value).longValue() * 1000L);
        }
        if (value instanceof String) {
            try {
                return LocalDateTime.parse((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        if (value instanceof Date) {
            return ((Date)value).toInstant().atZone(UTC).toLocalDateTime();
        }
        return null;
    }

    public static LocalTime parseLocalTime(Object value) {
        if (value == null) return null;
        if (value instanceof LocalTime) {
            return (LocalTime) value;
        }
        if (value instanceof Long && (long)value >= 0) {
            return timestamToLocaltime((Long) value);
        }
        if (value instanceof Integer && (Integer)value >= 0) {
            return timestamToLocaltime(((Integer) value).longValue() * 1000L);
        }
        if (value instanceof String) {
            try {
                return LocalTime.parse((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        if (value instanceof Time) {
            return ((Time)value).toLocalTime();
        }
        return null;
    }
}
