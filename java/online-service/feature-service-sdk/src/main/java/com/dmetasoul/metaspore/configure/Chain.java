package com.dmetasoul.metaspore.configure;

import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@Slf4j
@Data
public class Chain {
    private String name;
    private List<String> then;
    private List<String> when;

    private boolean isAny;

    private Long timeOut = 30000L;

    private TimeUnit timeOutUnit = TimeUnit.MILLISECONDS;

    private Map<String, Object> options;
    private List<TransformConfig> transforms;

    public void setThen(List<String> list) {
        then = list;
    }

    public void setThen(String str) {
        then = List.of(str);
    }

    public void setWhen(List<String> list) {
        when = list;
    }

    public void setWhen(String str) {
        when = List.of(str);
    }

    public Chain() {
    }

    public Chain(List<String> then, List<String> when, boolean isAny, Long timeOut, TimeUnit timeOutUnit) {
        this.then = then;
        this.when = when;
        this.isAny = isAny;
        this.timeOut = timeOut;
        this.timeOutUnit = timeOutUnit;
    }

    public Chain(List<String> then, List<String> when, boolean isAny) {
        this.then = then;
        this.when = when;
        this.isAny = isAny;
    }

    public Chain(List<String> then) {
        this.then = then;
    }

    public Chain(String taskName) {
        this.then = List.of(taskName);
    }

    public boolean isEmpty() {
        return CollectionUtils.isEmpty(then) && CollectionUtils.isEmpty(when);
    }

    public boolean noChanged(Chain chain) {
        if (chain == null) {
            return isEmpty();
        }
        if (CollectionUtils.isNotEmpty(then) && CollectionUtils.isNotEmpty(chain.getThen()) && then.size() != chain.getThen().size()) {
            return false;
        }
        if (CollectionUtils.isNotEmpty(when) && CollectionUtils.isNotEmpty(chain.getWhen()) && when.size() != chain.getWhen().size()) {
            return false;
        }
        if (CollectionUtils.isNotEmpty(when) != CollectionUtils.isNotEmpty(chain.getWhen())) {
            return false;
        }
        if (CollectionUtils.isNotEmpty(then) != CollectionUtils.isNotEmpty(chain.getThen())) {
            return false;
        }
        return true;
    }

    public void setTimeOutUnit(String data) {
        switch (StringUtils.capitalize(data.toLowerCase())) {
            case "Nanos":
                timeOutUnit = TimeUnit.NANOSECONDS;
                break;
            case "Micros":
                timeOutUnit = TimeUnit.MICROSECONDS;
                break;
            case "Millis":
                timeOutUnit = TimeUnit.MILLISECONDS;
                break;
            case "Seconds":
                timeOutUnit = TimeUnit.SECONDS;
                break;
            case "Minutes":
                timeOutUnit = TimeUnit.MINUTES;
                break;
            case "Hours":
                timeOutUnit = TimeUnit.HOURS;
                break;
            case "Days":
                timeOutUnit = TimeUnit.DAYS;
                break;
            default:
                log.warn("threadpool keepAliveTime timeunit default is seconds , config is {}", data);
                timeOutUnit = TimeUnit.SECONDS;
        }
    }

    public boolean checkAndDefault() {
        if (CollectionUtils.isEmpty(then) && CollectionUtils.isEmpty(when)) {
            log.error("the chain must has node in then or when!");
            return false;
        }
        Set<String> taskSet = Sets.newHashSet();
        int taskNum = 0;
        if (CollectionUtils.isNotEmpty(then)) {
            taskSet.addAll(then);
            taskNum += then.size();
        }
        if (CollectionUtils.isNotEmpty(when)) {
            taskSet.addAll(when);
            taskNum += when.size();
        }
        if (taskNum != taskSet.size()) {
            log.error("the chain has duplicate task in then or when!");
            return false;
        }
        setDefaultTimeOut(30000L, TimeUnit.MILLISECONDS);
        return true;
    }

    public void setDefaultTimeOut(long time, TimeUnit unit) {
        if (timeOut == null) {
            timeOut = time;
        }
        if (timeOutUnit == null) {
            timeOutUnit = unit;
        }
    }
}
