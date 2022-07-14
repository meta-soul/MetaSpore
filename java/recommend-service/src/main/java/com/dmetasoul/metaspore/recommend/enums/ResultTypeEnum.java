package com.dmetasoul.metaspore.recommend.enums;

public enum ResultTypeEnum {
    EMPTY(0,"empty"),
    VALUES(1, "values"),
    DATA(2,"data"),
    MILVUS(3, "milvus"),
    FEATUREARRAYS(4,"featureArrays"),
    FEATURETABLE(5,"featureTable"),

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
