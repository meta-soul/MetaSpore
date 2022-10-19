package com.dmetasoul.metaspore.configure;

import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;

@Slf4j
@Data
public class Source {
    private String name;
    // private String format;
    private String kind;
    private Map<String, Object> options;

    public String getKind() {
        if (StringUtils.isEmpty(kind)) return "request";
        return kind;
    }

    public boolean checkAndDefault() {
        if (StringUtils.isEmpty(name)) {
            log.error("source config name must not be empty!");
            return false;
        }
        if (options == null) {
            options = Maps.newHashMap();
        }
        if (getKind().equalsIgnoreCase("mongodb")) {
            if (!options.containsKey("uri") || !String.valueOf(options.get("uri")).startsWith("mongodb://")) {
                log.error("source mongodb config uri error!");
                return false;
            }
        }
        if (getKind().equalsIgnoreCase("jdbc")) {
            if (!options.containsKey("uri")) {
                log.error("source jdbc config uri must not be empty!");
                return false;
            }
            if (!options.containsKey("user")) options.put("user", "root");
            if (!options.containsKey("password")) options.put("password", "test");
            String uri = String.valueOf(options.containsKey("uri"));
            if (uri.startsWith("jdbc:mysql")) {
                if (!options.containsKey("driver")) options.put("driver", "com.mysql.cj.jdbc.Driver");
                else {
                    if (!String.valueOf(options.get("driver")).equals("com.mysql.cj.jdbc.Driver")) {
                        log.error("source jdbc mysql config driver must be com.mysql.cj.jdbc.Driver!");
                        return false;
                    }
                }
            }
        }
        if (getKind().equalsIgnoreCase("redis")) {
            if (!options.containsKey("standalone") && !options.containsKey("sentinel")
                    && !options.containsKey("cluster")) {
                options.put("standalone", Map.of("host", "localhost", "port", 6379));
            }
        }
        return true;
    }
}
