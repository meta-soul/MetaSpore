package com.dmetasoul.metaspore.recommend.configure;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.elasticsearch.common.Strings;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.concurrent.*;

@Slf4j
@Data
@Component
@ConfigurationProperties(prefix = "recommend.thread-pool")
public class ThreadPoolConfig {
    private ThreadPool feature;
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
            switch (Strings.capitalize(data.toLowerCase())) {
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
