package com.dmetasoul.metaspore.recommend.annotation;

import org.springframework.context.annotation.Scope;
import org.springframework.core.annotation.AliasFor;
import org.springframework.stereotype.Component;

import java.lang.annotation.*;

@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Component
@Scope("prototype")
public @interface DataServiceAnnotation {
    @AliasFor(
            annotation = Component.class
    )
    String value() default "";
}