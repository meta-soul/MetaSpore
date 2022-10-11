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
public enum JoinTypeEnum {
    INNER(0, "inner"),
    LEFT(1, "left"),
    RIGHT(2, "right"),
    FULL(3, "full");

    private Integer id;

    private String name;

    JoinTypeEnum(int id, String name) {
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

    public static JoinTypeEnum getEnumByName(String name) {
        for (JoinTypeEnum e : JoinTypeEnum.values()) {
            if (e.getName().equals(name)) {
                return e;
            }
        }
        log.warn("unknown join name :{}, default type is inner", name);
        return JoinTypeEnum.INNER;
    }

    public static JoinTypeEnum getEnumById(int id) {
        for (JoinTypeEnum e : JoinTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        log.warn("unknown join id :{}, default type is inner", id);
        return JoinTypeEnum.INNER;
    }
}
