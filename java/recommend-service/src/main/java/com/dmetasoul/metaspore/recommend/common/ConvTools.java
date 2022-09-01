package com.dmetasoul.metaspore.recommend.common;

import java.math.BigDecimal;
import java.sql.Time;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.Date;
import org.bson.types.Decimal128;

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

    public static BigDecimal parseBigDecimal(Object value) {
        if (value == null) return null;
        if (value instanceof BigDecimal) {
            return (BigDecimal) value;
        }
        if (value instanceof Decimal128) {
            return ((Decimal128)value).bigDecimalValue();
        }
        if (value instanceof Number) {
            return BigDecimal.valueOf(((Number) value).doubleValue());
        }
        if (value instanceof String) {
            try {
                return Decimal128.parse((String) value).bigDecimalValue();
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        return null;
    }

    public static String parseString(Object value) {
        if (value == null) return null;
        if (value instanceof String) {
            return (String) value;
        }
        try {
            return String.valueOf(value);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    public static Integer parseInteger(Object value) {
        if (value == null) return null;
        if (value instanceof Integer) {
            return (Integer) value;
        }
        if (value instanceof String) {
            try {
                return Integer.parseInt((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        return null;
    }

    public static Long parseLong(Object value) {
        if (value == null) return null;
        if (value instanceof Long) {
            return (Long) value;
        }
        if (value instanceof Integer) {
            return ((Integer) value).longValue();
        }
        if (value instanceof String) {
            try {
                return Long.parseLong((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        return null;
    }

    public static Double parseDouble(Object value) {
        if (value == null) return null;
        if (value instanceof Double) {
            return (Double) value;
        }
        if (value instanceof Number) {
            return ((Number)value).doubleValue();
        }
        if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        return null;
    }

    public static Float parseFloat(Object value) {
        if (value == null) return null;
        if (value instanceof Float) {
            return (Float) value;
        }
        // mybe diff
        if (value instanceof Number) {
            return ((Number)value).floatValue();
        }
        if (value instanceof String) {
            try {
                return Float.parseFloat((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        return null;
    }

    public static Boolean parseBoolean(Object value) {
        if (value == null) return null;
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof Integer) {
            return (int)value != 0;
        }
        if (value instanceof String) {
            try {
                return Boolean.parseBoolean((String) value);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        return null;
    }
}
