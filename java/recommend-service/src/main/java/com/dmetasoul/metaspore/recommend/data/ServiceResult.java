package com.dmetasoul.metaspore.recommend.data;

import lombok.Data;

@Data
public class ServiceResult {
    private ServiceStatus code;
    private String msg;
    private DataResult data;
    private String id;

    public enum ServiceStatus {
        UNKNOWN(-10,"unknown"),
        FAIL(-1,"fail"),
        SUCCESS(0,"success");

        private Integer id;

        private String name;

        ServiceStatus(int id, String name){
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

        public static ServiceStatus of(int code) {
            for (ServiceStatus status : ServiceStatus.values()) {
                if (status.getId() == code) {
                    return status;
                }
            }
            return ServiceStatus.UNKNOWN;
        }
    }

    private ServiceResult(ServiceStatus code, String msg, DataResult data, String id) {
        this.code = code;
        this.msg = msg;
        this.data = data;
        this.id = id;
    }

    public static ServiceResult of(int code, String msg) {
        return ServiceResult.of(code, msg, null, null);
    }

    public static ServiceResult of(DataResult data) {
        return new ServiceResult(ServiceStatus.SUCCESS, "success!", data, null);
    }

    public static ServiceResult of(DataResult data, String id) {
        ServiceResult instacnce = ServiceResult.of(data);
        instacnce.id = id;
        return instacnce;
    }

    public static ServiceResult of(int code, String msg, DataResult data) {
        return new ServiceResult(ServiceStatus.of(code), msg, data, null);
    }

    public static ServiceResult of(int code, String msg, DataResult data, String id) {
        return new ServiceResult(ServiceStatus.of(code), msg, data, id);
    }
}
