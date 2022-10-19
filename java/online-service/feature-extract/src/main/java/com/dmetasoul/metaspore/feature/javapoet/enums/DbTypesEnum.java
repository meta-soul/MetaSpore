//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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