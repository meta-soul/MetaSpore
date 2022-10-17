package com.dmetasoul.metaspore;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@Data
public class UDFLoader {
    private URLClassLoader classLoader;
    private Set<URL> urls;
    private Map<String, Class<?>> udfClassMap;
    private Map<String, Class<?>> udfNoCaseClassMap;

    public UDFLoader() {
        this.urls = Sets.newHashSet();
        this.udfClassMap = Maps.newHashMap();
        this.udfNoCaseClassMap = Maps.newHashMap();
    }

    public void addUrls(Collection<URL> urls) {
        this.urls.addAll(urls);
    }

    @SneakyThrows
    public void close() {
        if (null != classLoader) {
            classLoader.close();
        }
    }

    public void registerUDF(List<Map<String, String>> info) {
        if (udfClassMap == null) {
            udfClassMap = Maps.newHashMap();
        }
        if (CollectionUtils.isEmpty(urls)) {
            log.warn("udf load url is empty!");
            return;
        }
        if (classLoader == null) {
            this.classLoader = new URLClassLoader(urls.toArray(URL[]::new));
        }
        if (CollectionUtils.isNotEmpty(info)) {
            info.forEach(item -> item.forEach((name, className) -> {
                try {
                    Class<?> cls = classLoader.loadClass(className);
                    if (udfClassMap.containsKey(name)) {
                        log.warn("udf function found duplicate class: {}", name);
                    }
                    udfClassMap.put(name, cls);
                    udfNoCaseClassMap.put(name.toLowerCase(), cls);
                } catch (ClassNotFoundException e) {
                    log.error("udf class not found at name: {}, className: {}", name, className);
                    throw new RuntimeException(e);
                }
            }));
        }
    }

    @SneakyThrows
    @SuppressWarnings("unchecked")
    public <T> T getBean(String name, Class<?> cls, boolean ignoreCase) {
        Class<?> cla = getCls(name, cls, ignoreCase);
        if (cla != null) {
            if (cls == null || cls.isAssignableFrom(cla)) {
                Object function = cla.getConstructor().newInstance();
                log.info("load bean: {} from udf", name);
                return (T) function;
            }
        }
        return null;
    }

    public Class<?> getCls(String name, Class<?> cls, boolean ignoreCase) {
        if (udfClassMap != null && udfClassMap.containsKey(name)) {
            Class<?> cla = udfClassMap.get(name);
            if (cls == null || cls.isAssignableFrom(cla)) {
                return cla;
            }
        }
        if (ignoreCase && udfNoCaseClassMap != null && udfNoCaseClassMap.containsKey(name)) {
            Class<?> cla = udfNoCaseClassMap.get(name);
            if (cls == null || cls.isAssignableFrom(cla)) {
                return cla;
            }
        }
        try {
            if (classLoader == null) {
                this.classLoader = new URLClassLoader(urls.toArray(URL[]::new));
            }
            Class<?> cla = classLoader.loadClass(name);
            if (cla == null) {
                return null;
            }
            if (cls == null || cls.isAssignableFrom(cla)) {
                udfClassMap.put(name, cla);
            }
        } catch (ClassNotFoundException e) {
            log.warn("udf class not found at name: {}", name);
        }
        return null;
    }

    @SneakyThrows
    public void addJarURL(@NonNull String filePath) {
        if (urls == null) {
            urls = Sets.newHashSet();
        }
        File jarFile = new File(filePath);
        if (!jarFile.exists()) {
            throw new IllegalStateException("path + jar path is not exist! Path: " + filePath);
        }
        int urlNum = urls.size();
        urls.add(jarFile.toURI().toURL());
        if (urlNum < urls.size() && classLoader != null) {
            classLoader.close();
            this.classLoader = new URLClassLoader(urls.toArray(URL[]::new));
        }
    }

    @SneakyThrows
    public void addJarURL(@NonNull String path, @NonNull String name) {
        Path filePath = Paths.get(Paths.get(path).toString(), name);
        addJarURL(filePath.toString());
    }
}