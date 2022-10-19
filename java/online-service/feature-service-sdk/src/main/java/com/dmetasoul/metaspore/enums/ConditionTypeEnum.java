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
package com.dmetasoul.metaspore.enums;

import lombok.extern.slf4j.Slf4j;

import java.util.Collection;

@Slf4j
public enum ConditionTypeEnum {
    GT(0, "gt", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 != null) {
                if (value2 == null) return true;
                if (value1 instanceof Comparable && value1.getClass().isAssignableFrom(value2.getClass())) {
                    return ((Comparable) value1).compareTo(value2) > 0;
                }
            }
            return false;
        }
    }),
    GE(1, "ge", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 == null && value2 == null) return true;
            if (value1 != null) {
                if (value2 == null) return true;
                if (value1 instanceof Comparable && value1.getClass().isAssignableFrom(value2.getClass())) {
                    return ((Comparable) value1).compareTo(value2) >= 0;
                }
            }
            return false;
        }
    }),
    LT(2, "lt", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value2 != null) {
                if (value1 == null) return true;
                if (value1 instanceof Comparable && value1.getClass().isAssignableFrom(value2.getClass())) {
                    return ((Comparable) value1).compareTo(value2) < 0;
                }
            }
            return false;
        }
    }),
    LE(3, "le", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 == null && value2 == null) return true;
            if (value2 != null) {
                if (value1 == null) return true;
                if (value1 instanceof Comparable && value1.getClass().isAssignableFrom(value2.getClass())) {
                    return ((Comparable) value1).compareTo(value2) <= 0;
                }
            }
            return false;
        }
    }),
    NE(4, "ne", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 == null && value2 == null) return false;
            return value1 == null || !value1.equals(value2);
        }
    }),
    EQ(5, "eq", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 == null && value2 == null) return true;
            return value1 != null && value1.equals(value2);
        }
    }),
    IN(6, "in", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 == null && value2 == null) return true;
            if (value2 != null) {
                if (value2 instanceof Collection) {
                    return ((Collection<?>) value2).contains(value1);
                }
                return value2.equals(value1);
            }
            return false;
        }
    }),
    NIN(7, "nin", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            if (value1 == null && value2 == null) return false;
            if (value2 != null) {
                if (value2 instanceof Collection) {
                    return !((Collection<?>) value2).contains(value1);
                }
                return !value2.equals(value1);
            }
            return true;
        }
    }),
    UNKNOWN(20, "unknown", new Op() {
        @Override
        public boolean op(Object value1, Object value2) {
            return false;
        }
    });

    private Integer id;

    private String name;

    private Op op;

    ConditionTypeEnum(int id, String name, Op op) {
        this.id = id;
        this.name = name;
        this.op = op;
    }

    public int getId() {
        return id;
    }

    public boolean Op(Object value1, Object value2) {
        return this.op.op(value1, value2);
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

    public static abstract class Op {
        public abstract boolean op(Object value1, Object value2);
    }
}
