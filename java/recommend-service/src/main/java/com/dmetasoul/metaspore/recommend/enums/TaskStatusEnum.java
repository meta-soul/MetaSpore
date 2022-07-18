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

public enum TaskStatusEnum {
    INIT(0,"init"),
    CHECK_FAIL(1, "check_fail"),
    EXEC_FAIL(2,"exec_fail"),
    RESULT_ERROR(3, "result_error"),
    DEPEND_INIT_FAIL(4,"depend_init_fail"),
    DEPEND_EXEC_FAIL(5,"depend_exec_fail"),

    SUCCESS(20,"success");

    private Integer id;

    private String name;

    TaskStatusEnum(int id, String name){
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

    public static TaskStatusEnum getEnumByName(String name) {
        for (TaskStatusEnum e : TaskStatusEnum.values()) {
            if (e.getName().equals(name)) {
                return e;
            }
        }
        return null;
    }

    public static TaskStatusEnum getEnumById(int id) {
        for (TaskStatusEnum e : TaskStatusEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        return null;
    }
}
