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