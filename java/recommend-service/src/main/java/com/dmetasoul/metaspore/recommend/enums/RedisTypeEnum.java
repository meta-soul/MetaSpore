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
package com.dmetasoul.metaspore.recommend.enums;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public enum ConditionTypeEnum {
    GT(0,"gt"),
    GE(1, "ge"),
    LT(2,"lt"),
    LE(3, "le"),
    NE(4,"ne"),
    EQ(5, "eq"),
    IN(6,"in"),
    NIN(7, "nin"),
    UNKNOWN(20, "unknown");

    private Integer id;

    private String name;

    ConditionTypeEnum(int id, String name){
        this.id = id;
        this.name = name;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public static ConditionTypeEnum getEnumByName(String name) {
        for (ConditionTypeEnum e : ConditionTypeEnum.values()) {
            if (e.getName().equals(name.strip().toLowerCase())) {
                return e;
            }
        }
        log.warn("unknown condition name :{}, default type is unknown", name);
        return ConditionTypeEnum.UNKNOWN;
    }

    public static ConditionTypeEnum getEnumById(int id) {
        for (ConditionTypeEnum e : ConditionTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        log.warn("unknown condition id :{}, default type is unknown", id);
        return ConditionTypeEnum.UNKNOWN;
    }
}
