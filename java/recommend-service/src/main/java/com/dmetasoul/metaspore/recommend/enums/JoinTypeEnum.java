package com.dmetasoul.metaspore.recommend.enums;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public enum JoinTypeEnum {
    INNER(0,"inner"),
    LEFT(1, "left"),
    RIGHT(2,"right"),
    FULL(3, "full");

    private Integer id;

    private String name;

    JoinTypeEnum(int id, String name){
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
