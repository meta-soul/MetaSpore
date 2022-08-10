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

package com.dmetasoul.metaspore.recommend.common;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;

import java.lang.annotation.Annotation;
import java.util.Map;

/**
 * 用于获取spring bean 工具类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Component
public class SpringBeanUtil implements ApplicationContextAware {

    private static ApplicationContext ctx;

    @Override
    public void setApplicationContext(@SuppressWarnings("NullableProblems") ApplicationContext applicationContext) throws BeansException {
        SpringBeanUtil.ctx = applicationContext;
    }

    public static ApplicationContext getApplicationContext() {
        return ctx;
    }

    //通过name获取 Bean. 默认首字母大写
    public static Object getBean(String name) {
        try {
            return getApplicationContext().getBean(StringUtils.capitalize(name));
        } catch (BeansException ex) {
            // log.warn("spring get bean by name {} fail", name);
            return null;
        }
    }
    public static Object getBeanByName(String name) {
        try {
            return getApplicationContext().getBean(name);
        } catch (BeansException ex) {
            // log.warn("spring get bean by name {} fail", name);
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
