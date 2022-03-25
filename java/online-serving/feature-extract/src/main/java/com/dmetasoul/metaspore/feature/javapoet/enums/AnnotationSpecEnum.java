package com.dmetasoul.metaspore.feature.javapoet.enums;

import com.squareup.javapoet.AnnotationSpec;
import com.squareup.javapoet.ClassName;

// jpa 注解引用定义
public enum AnnotationSpecEnum {
    AUTOWIRED,
    OVERRIDE,
    RESTCONTROLLER,
    PATHVARIABLE,
    REQUESTBODY;

    public static AnnotationSpec getAnnotationSpec(AnnotationSpecEnum type) {
        switch (type) {
            case AUTOWIRED:
                return AnnotationSpec.builder(ClassName.get("org.springframework.beans.factory.annotation", "Autowired")).build();
            case OVERRIDE:
                return AnnotationSpec.builder(Override.class).build();
            case RESTCONTROLLER:
                return AnnotationSpec.builder(ClassName.get("org.springframework.web.bind.annotation", "RestController")).build();
            case PATHVARIABLE:
                return AnnotationSpec.builder(ClassName.get("org.springframework.beans.factory.annotation", "PathVariable")).build();
            case REQUESTBODY:
                return AnnotationSpec.builder(ClassName.get("org.springframework.web.bind.annotation", "RequestBody")).build();
            default:
                break;
        }
        return null;
    }
}
