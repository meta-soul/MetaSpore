package com.dmetasoul.metaspore;

import com.dmetasoul.metaspore.annotation.FeatureAnnotation;
import com.dmetasoul.metaspore.dataservice.DataService;
import com.dmetasoul.metaspore.datasource.DataSource;
import com.dmetasoul.metaspore.functions.Function;
import com.dmetasoul.metaspore.relyservice.RelyService;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.reflections.Reflections;

import java.util.Map;
import java.util.Set;

@Data
@Slf4j
public class FeatureServiceManager implements AutoCloseable {
    /**
     * 用于注册保存各种DataService，每次更新配置，重新生成；
     */
    private Map<String, DataService> dataServices;
    /**
     * 用于注册保存各种DataSource，每次更新配置，类似于dataServices
     */
    private Map<String, DataSource> dataSources;
    /**
     * 用于注册保存各种Function，每次更新配置，类似于dataServices
     */
    private Map<String, Object> beanMap;

    private Set<String> packages;

    private Map<String, Class<?>> featureClassMap;

    private Map<String, Class<?>> featureLowerClassMap;

    private Map<String, RelyService> relyServices;

    private UDFLoader udfLoader;

    public FeatureServiceManager() {
        this.packages = Sets.newHashSet();
        this.packages.add(this.getClass().getPackageName());
        featureClassMap = Maps.newHashMap();
        featureLowerClassMap = Maps.newHashMap();
        udfLoader = new UDFLoader();
        beanMap = Maps.newHashMap();
        relyServices = Maps.newHashMap();
    }

    public void addPackage(String packageName) {
        this.packages.add(packageName);
    }

    @SuppressWarnings("unchecked")
    public void scanSubClass(Reflections reflection, Class<?> cls, Map<String, Class<?>> featureClassMap) {
        Set<Class<?>> subClassSet = reflection.getSubTypesOf((Class<Object>) cls);
        for (Class<?> cla : subClassSet) {
            if (featureClassMap.containsKey(cla.getName())) {
                log.warn("Found duplicate feature name: {} class in scanSubClass", cla.getName());
            }
            featureClassMap.put(cla.getName(), cla);
        }
    }

    public void scanClass() {
        for (String packageName : this.packages) {
            Reflections reflection = new Reflections(packageName);
            Set<Class<?>> featureClassSet = reflection.getTypesAnnotatedWith(FeatureAnnotation.class);
            for (Class<?> cls : featureClassSet) {
                if (cls.getAnnotation(FeatureAnnotation.class) == null) {
                    continue;
                }
                String annotationValue = cls.getAnnotation(FeatureAnnotation.class).value();
                if (StringUtils.isEmpty(annotationValue)) {
                    continue;
                }
                if (featureClassMap.containsKey(annotationValue)) {
                    log.warn("Found duplicate feature name: {} class in {}", annotationValue, packageName);
                }
                featureClassMap.put(annotationValue, cls);
                featureLowerClassMap.put(annotationValue.toLowerCase(), cls);
            }
            scanSubClass(reflection, DataSource.class, featureClassMap);
            scanSubClass(reflection, DataService.class, featureClassMap);
            scanSubClass(reflection, Function.class, featureClassMap);
        }
    }

    public DataService getDataService(String name) {
        if (MapUtils.isNotEmpty(dataServices)) {
            return dataServices.get(name);
        }
        return null;
    }

    @SneakyThrows
    public Function getFunction(String name) {
        if (StringUtils.isEmpty(name)) {
            return null;
        }
        return getBean(name, Function.class, true, true);
    }

    public Map<String, DataSource> getDataSources() {
        if (MapUtils.isNotEmpty(dataSources)) {
            return dataSources;
        }
        return Map.of();
    }

    public DataSource getDataSource(String name) {
        if (MapUtils.isNotEmpty(dataSources)) {
            return dataSources.get(name);
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    public <T> T getRelyService(String name, Class<?> cls) {
        if (relyServices == null) relyServices = Maps.newHashMap();
        RelyService relyService = relyServices.get(name);
        if (relyService == null) {
            return null;
        }
        if (!relyService.getClass().equals(cls)) {
            log.warn("load rely service:{} type:{} is not match {}", name, relyService.getClass(), cls);
            return null;
        }
        log.info("get rely service:{}", name);
        return (T) relyServices;
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public <T> T getRelyServiceOrSet(String name, Class<?> cls, Map<String, Object> option) {
        if (relyServices == null) relyServices = Maps.newHashMap();
        RelyService relyService = relyServices.get(name);
        if (relyService == null) {
            relyService = (RelyService) cls.getConstructor().newInstance();
            relyService.init(option);
            relyServices.put(name, relyService);
        }
        if (!relyService.getClass().equals(cls)) {
            log.error("load rely service:{} type:{} is not match {}", name, relyService.getClass(), cls);
            throw new RuntimeException("load rely service type mot match at " + name);
        }
        log.info("load rely service:{}", name);
        return (T) relyService;
    }

    @SneakyThrows
    @SuppressWarnings("unchecked")
    public <T> T getBean(String name, Class<?> cls, boolean hold, boolean ignoreCase) {
        if (hold && MapUtils.isNotEmpty(beanMap) && beanMap.containsKey(name)) {
            log.info("load bean: {} from beanMap", name);
            return (T) beanMap.get(name);
        }
        Class<?> cla = null;
        if (featureClassMap.containsKey(name)) {
            cla = featureClassMap.get(name);
        } else if (ignoreCase && featureLowerClassMap.containsKey(name.toLowerCase())) {
            cla = featureLowerClassMap.get(name.toLowerCase());
        } else if (cls != null && featureClassMap.containsKey(cls.getName())) {
            cla = featureClassMap.get(cls.getName());
        }
        if (cla != null) {
            if (cls == null || cls.isAssignableFrom(cla)) {
                log.info("load bean: {} from feature service", name);
                Object bean = cla.getConstructor().newInstance();
                if (hold) {
                    this.beanMap.put(name, bean);
                }
                return (T) bean;
            }
        } else {
            Object bean = udfLoader.getBean(name, cls, ignoreCase);
            if (hold && bean != null) {
                this.beanMap.put(name, bean);
            }
            return (T) bean;
        }
        return null;
    }

    public void addDataSource(String name, DataSource dataSource) {
        if (StringUtils.isEmpty(name) || dataSource == null) {
            return;
        }
        if (!dataSource.isInit()) {
            log.warn("the datasource: {} is not init!", name);
            return;
        }
        if (dataSources == null) {
            dataSources = Maps.newHashMap();
        }
        dataSources.put(name, dataSource);
    }

    public void addDataService(String name, DataService dataService) {
        if (StringUtils.isEmpty(name) || dataService == null) {
            return;
        }
        if (!dataService.isInit()) {
            log.warn("the dataService: {} is not init!", name);
            return;
        }
        if (dataServices == null) {
            dataServices = Maps.newHashMap();
        }
        dataServices.put(name, dataService);
    }

    @Override
    public void close() throws Exception {
        if (MapUtils.isNotEmpty(dataSources)) {
            dataSources.forEach((name, source) -> source.close());
            dataSources.clear();
        }
        if (MapUtils.isNotEmpty(dataServices)) {
            dataServices.forEach((name, service) -> service.close());
            dataServices.clear();
        }
        udfLoader.close();
        relyServices.forEach((name, service) -> service.close());
    }
}
