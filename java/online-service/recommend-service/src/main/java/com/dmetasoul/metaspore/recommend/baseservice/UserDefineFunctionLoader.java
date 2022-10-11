package com.dmetasoul.metaspore.recommend.baseservice;

import com.dmetasoul.metaspore.recommend.bucketizer.LayerBucketizer;
import com.dmetasoul.metaspore.recommend.configure.FunctionConfig;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.MergeOperator;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.TransformFunction;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.UpdateOperator;
import com.google.common.collect.Maps;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

@Slf4j
@Component
public class UserDefineFunctionLoader {
    private URLClassLoader classLoader;

    @Autowired
    private FunctionConfig functionConfig;
    private Map<String, Function> fieldFunctions;
    private Map<String, TransformFunction> transformFunctions;
    private Map<String, MergeOperator> transformMergeOperators;
    private Map<String, UpdateOperator> transformUpdateOperators;
    private Map<String, LayerBucketizer> layerBucketizers;

    @SneakyThrows
    public void init() {
        log.info("functionConfig: {}", functionConfig);
        this.fieldFunctions = Maps.newHashMap();
        this.transformFunctions = Maps.newHashMap();
        this.transformMergeOperators = Maps.newHashMap();
        this.transformUpdateOperators = Maps.newHashMap();
        this.layerBucketizers = Maps.newHashMap();
        if (functionConfig != null && functionConfig.getPath() != null
                && CollectionUtils.isNotEmpty(functionConfig.getJars())) {
            URL[] urls = new URL[functionConfig.getJars().size()];
            for (int i = 0; i < functionConfig.getJars().size(); ++i) {
                FunctionConfig.JarInfo jarInfo = functionConfig.getJars().get(i);
                urls[i] = getJarURL(functionConfig.getPath(), jarInfo);
            }
            this.classLoader = new URLClassLoader(urls);
            for (FunctionConfig.JarInfo jarInfo : functionConfig.getJars()) {
                loadJarInfo(functionConfig.getPath(), jarInfo);
            }
        }
    }

    @SneakyThrows
    public void close() {
        if (null != classLoader) {
            classLoader.close();
        }
    }

    @SuppressWarnings("unchecked")
    public <T> T getBean(String name, Class<?> cls) {
        if (cls.equals(Function.class) && fieldFunctions.containsKey(name)) {
            return (T) fieldFunctions.get(name);
        }
        if (cls.equals(TransformFunction.class) && transformFunctions.containsKey(name)) {
            return (T) transformFunctions.get(name);
        }
        if (cls.equals(MergeOperator.class) && transformMergeOperators.containsKey(name)) {
            return (T) transformMergeOperators.get(name);
        }
        if (cls.equals(UpdateOperator.class) && transformUpdateOperators.containsKey(name)) {
            return (T) transformUpdateOperators.get(name);
        }
        if (cls.equals(LayerBucketizer.class) && layerBucketizers.containsKey(name)) {
            return (T) layerBucketizers.get(name);
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    public <T> void registerUDF(List<Map<String, String>> info, @NonNull Map<String, T> beans, Class<?> cla) {
        if (CollectionUtils.isNotEmpty(info)) {
            info.forEach(item -> {
                item.forEach((name, className) -> {
                    try {
                        Class<?> cls = classLoader.loadClass(className);
                        Object bean = cls.getConstructor().newInstance();
                        if (cla.isAssignableFrom(bean.getClass())) {
                            beans.put(name, (T) bean);
                        } else {
                            log.error("bean class not match at name: {}, className: {}, class: {}", name, className, bean.getClass());
                            throw new IllegalStateException("bean class not match at " + name);
                        }
                    } catch (ClassNotFoundException e) {
                        log.error("udf class not found at name: {}, className: {}", name, className);
                        throw new RuntimeException(e);
                    } catch (NoSuchMethodException | InvocationTargetException | InstantiationException |
                            IllegalAccessException e) {
                        log.error("udf class instance create fail at name: {}, className: {}", name, className);
                        throw new RuntimeException(e);
                    }
                });
            });

        }
    }

    @SneakyThrows
    public URL getJarURL(String path, @NonNull FunctionConfig.JarInfo jarInfo) {
        Path filePath = Paths.get(Paths.get(path).toString(), jarInfo.getName());
        File jarFile = new File(filePath.toString());
        if (!jarFile.exists()) {
            throw new IllegalStateException("path + jar path is not exist! Path: " + filePath);
        }
        return jarFile.toURI().toURL();
    }

    public void loadJarInfo(String path, @NonNull FunctionConfig.JarInfo jarInfo) {
        if (classLoader == null) {
            throw new IllegalStateException("UserDefineFunctionLoader need init first");
        }
        registerUDF(jarInfo.getFieldFunction(), fieldFunctions, Function.class);
        registerUDF(jarInfo.getTransformFunction(), transformFunctions, TransformFunction.class);
        registerUDF(jarInfo.getTransformMergeOperator(), transformMergeOperators, MergeOperator.class);
        registerUDF(jarInfo.getTransformUpdateOperator(), transformUpdateOperators, UpdateOperator.class);
        registerUDF(jarInfo.getLayerBucketizer(), layerBucketizers, LayerBucketizer.class);
    }
}