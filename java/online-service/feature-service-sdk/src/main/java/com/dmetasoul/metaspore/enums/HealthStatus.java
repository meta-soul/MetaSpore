package com.dmetasoul.metaspore.enums;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public enum HealthStatus {
    UP(0, "UP"),
    DOWN(1, "DOWN"),
    OUT_OF_SERVICE(2, "OUT_OF_SERVICE"),
    UNKNOWN(3, "UNKNOWN");

    private Integer id;

    private String name;

    HealthStatus(int id, String name) {
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

    public static HealthStatus getEnumByName(String name) {
        for (HealthStatus e : HealthStatus.values()) {
            if (e.getName().equals(name)) {
                return e;
            }
        }
        log.warn("unknown status name :{}, default type is inner", name);
        return HealthStatus.UNKNOWN;
    }

    public static HealthStatus getEnumById(int id) {
        for (HealthStatus e : HealthStatus.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        log.warn("unknown status id :{}, default type is inner", id);
        return HealthStatus.UNKNOWN;
    }
}
