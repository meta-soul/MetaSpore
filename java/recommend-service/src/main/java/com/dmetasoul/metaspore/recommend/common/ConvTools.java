package com.dmetasoul.metaspore.recommend.common;

import java.sql.Timestamp;
import java.time.LocalDateTime;

public class ConvTools {
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
}
