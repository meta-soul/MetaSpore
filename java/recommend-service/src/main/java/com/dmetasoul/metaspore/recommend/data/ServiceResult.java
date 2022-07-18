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
package com.dmetasoul.metaspore.recommend.data;

import lombok.Data;
/**
 * 用于restfull api接口输出
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
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
