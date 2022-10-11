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
package com.dmetasoul.metaspore.recommend.configure;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.concurrent.*;

/**
 * 线程池配置类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
@Component
@ConfigurationProperties(prefix = "recommend.thread-pool")
public class ThreadPoolConfig {
    private ThreadPool source;
    private ThreadPool task;
    private ThreadPool workflow;

    @Data
    public static class ThreadPool {
        private int corePoolSize;
        private int maximumPoolSize;
        private int keepAliveTime;
        private TimeUnit timeunit;
        private int capacity;

        public void setTimeunit(String data) {
            switch (StringUtils.capitalize(data.toLowerCase())) {
                case "Nanos":
                    timeunit = TimeUnit.NANOSECONDS;
                    break;
                case "Micros":
                    timeunit = TimeUnit.MICROSECONDS;
                    break;
                case "Millis":
                    timeunit = TimeUnit.MILLISECONDS;
                    break;
                case "Seconds":
                    timeunit = TimeUnit.SECONDS;
                    break;
                case "Minutes":
                    timeunit = TimeUnit.MINUTES;
                    break;
                case "Hours":
                    timeunit = TimeUnit.HOURS;
                    break;
                case "Days":
                    timeunit = TimeUnit.DAYS;
                    break;
                default:
                    log.warn("threadpool keepAliveTime timeunit default is seconds , config is {}", data);
                    timeunit = TimeUnit.SECONDS;
            }
        }
    }
}
