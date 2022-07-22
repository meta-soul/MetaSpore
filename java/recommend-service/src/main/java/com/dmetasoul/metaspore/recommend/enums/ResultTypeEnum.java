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

public enum ResultTypeEnum {
    EMPTY(0,"empty"),
    VALUES(1, "values"),
    DATA(2,"data"),
    MILVUS(3, "milvus"),
    FEATUREARRAYS(4,"featureArrays"),
    FEATURETABLE(5,"featureTable"),

    PREDICTRESULT(6,"predictResult"),

    EXCEPTION(22,"exception");

    private Integer id;

    private String name;

    ResultTypeEnum(int id, String name){
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

    public static ResultTypeEnum getEnumByName(String name) {
        for (ResultTypeEnum e : ResultTypeEnum.values()) {
            if (e.getName().equals(name)) {
                return e;
            }
        }
        return null;
    }

    public static ResultTypeEnum getEnumById(int id) {
        for (ResultTypeEnum e : ResultTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        return null;
    }
}
