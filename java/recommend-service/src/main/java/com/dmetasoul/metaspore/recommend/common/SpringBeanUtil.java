package com.dmetasoul.metaspore.recommend.common;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;

import java.lang.annotation.Annotation;
import java.util.Map;

@Slf4j
@Component
public class SpringBeanUtil implements ApplicationContextAware {

    private static ApplicationContext ctx;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        SpringBeanUtil.ctx = applicationContext;
    }

    public static ApplicationContext getApplicationContext() {
        return ctx;
    }

    //通过name获取 Bean.
    public static Object getBean(String name) {
        try {
            return getApplicationContext().getBean(name);
        } catch (BeansException ex) {
            log.warn("spring get bean by name {} fail", name);
            return null;
        }
    }

    //通过class获取Bean.
    public static <T> T getBean(Class<T> clazz) {
        try {
            return getApplicationContext().getBean(clazz);
        } catch (BeansException ex) {
            log.warn("spring get bean by class {} fail", clazz);
            return null;
        }
    }

    //通过name,以及Clazz返回指定的Bean
    public static <T> T getBean(String name, Class<T> clazz) {
        try {
            return getApplicationContext().getBean(name, clazz);
        } catch (BeansException ex) {
            log.warn("spring get bean by name, class {} fail", name);
            return null;
        }
    }

    public static Map<String, Object> getBeanMapByAnnotation(Class<? extends Annotation> clazz) {
        return ctx.getBeansWithAnnotation(clazz);
    }

}
