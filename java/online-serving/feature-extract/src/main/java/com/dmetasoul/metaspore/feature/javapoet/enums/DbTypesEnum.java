package com.dmetasoul.metaspore.feature.javapoet.enums;

import java.util.Objects;

// db 类型定义
public enum DbTypesEnum {
    MONGODB("mongodb"),
    MYSQL("mysql");

    private final String dbType;

    DbTypesEnum(String dbType) {
        this.dbType = dbType;
    }

    public String getDbType() {
        return dbType;
    }

    public static DbTypesEnum valueof(String dbType) throws Exception {
        DbTypesEnum[] values = DbTypesEnum.values();
        for (DbTypesEnum value : values) {
            if (Objects.equals(value.getDbType(), dbType)) {
                return value;
            }
        }
        throw new Exception("Unmatched dbType: " + dbType);
    }
}
