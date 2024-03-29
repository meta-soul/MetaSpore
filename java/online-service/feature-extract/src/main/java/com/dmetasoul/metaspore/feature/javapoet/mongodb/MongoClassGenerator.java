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

package com.dmetasoul.metaspore.feature.javapoet.mongodb;

import com.squareup.javapoet.*;
import com.dmetasoul.metaspore.feature.dao.TableAttributes;
import com.dmetasoul.metaspore.feature.javapoet.BaseClassGenerator;
import com.dmetasoul.metaspore.feature.javapoet.PackageInfo;
import com.dmetasoul.metaspore.feature.javapoet.enums.AnnotationSpecEnum;
import com.dmetasoul.metaspore.feature.utils.GeneratorUtil;
import org.springframework.data.annotation.Id;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.lang.model.element.Modifier;
import java.util.Collection;
import java.util.Optional;

public class MongoClassGenerator implements BaseClassGenerator {

    /**
     * @param table
     * @param packageInfo
     * @return
     */
    @Override
    public JavaFile generateDomain(TableAttributes table, PackageInfo packageInfo) {
        AnnotationSpec annotationSpec = AnnotationSpec.builder(Document.class)
                .addMember("collection",
                        CodeBlock.builder()
                                .add("$S", table.getCollectionName())
                                .build()
                ).build();

        FieldSpec fileSpec = FieldSpec.builder(ClassName.get(String.class), "Id", Modifier.PRIVATE)
                .addAnnotation(Id.class)
                .build();

        TypeSpec typeSpec = TypeSpec.classBuilder(GeneratorUtil.toCamelCase(table.getTableName(), true))
                .addJavadoc("Automatically Generated Code! Do not Modify this file")
                .addModifiers(Modifier.PUBLIC)
                .addAnnotations(GeneratorUtil.entityAnnotations())
                .addAnnotation(annotationSpec)
                .addField(fileSpec)
                .addFields(GeneratorUtil.entityFields(table.getColumns()))
                .build();

        return GeneratorUtil.javaFile(packageInfo.getDomain(), typeSpec);
    }


    /**
     * @param repoPrefixName
     * @param packageInfo
     * @param daoClass
     * @return
     */
    @Override
    public JavaFile generateRepository(String repoPrefixName, PackageInfo packageInfo,
                                       JavaFile daoClass, TableAttributes table) {
        ParameterizedTypeName parameterizedTypeName = ParameterizedTypeName.get(
                ClassName.bestGuess("org.springframework.data.mongodb.repository.MongoRepository")
                , GeneratorUtil.getClassName(daoClass)
                , ClassName.get(String.class)
        );

        MethodSpec findByIdMethod = MethodSpec.methodBuilder("findByQueryid")
                .addModifiers(Modifier.PUBLIC, Modifier.ABSTRACT)
                .addParameter(ParameterSpec.builder(String.class, "queryid").build())
                .returns(
                        ParameterizedTypeName.get(ClassName.get(Optional.class)
                                , ClassName.get(packageInfo.getDomain(), GeneratorUtil.toCamelCase(table.getTableName(), true)))
                )
                .build();

        MethodSpec findByIdsMethod = MethodSpec.methodBuilder("findByQueryidIn")
                .addModifiers(Modifier.PUBLIC, Modifier.ABSTRACT)
                .addParameter(ParameterSpec.builder(Collection.class, "queryid").build())
                .returns(
                        ParameterizedTypeName.get(
                                ClassName.get(Collection.class)
                                , GeneratorUtil.getClassName(daoClass)
                                //ClassName.get(PathEnum.DOMAIN.getPath(), GeneratorUtil.toCamelCase(table.getTableName(), true))
                        )
                )
                .build();

        TypeSpec typeSpec = TypeSpec.interfaceBuilder(GeneratorUtil.toCamelCase(repoPrefixName, true) + "Repository")
                .addJavadoc("Automatically Generated Code! Do not Modify this file")
                .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
                .addAnnotation(ClassName.bestGuess("org.springframework.stereotype.Repository"))
                .addSuperinterface(parameterizedTypeName)
                .addMethod(findByIdMethod)
                .addMethod(findByIdsMethod)
                .build();


        return GeneratorUtil.javaFile(packageInfo.getRepository(), typeSpec);
    }


    public static JavaFile generateMongoController(String name, PackageInfo packageInfo, JavaFile entityClass, JavaFile repositoryClass, TableAttributes table) {
        MethodSpec addMethod = MethodSpec.methodBuilder("addNew" + GeneratorUtil.toCamelCase(table.getTableName(), true))
                .addAnnotation(GeneratorUtil.controllerMapping("PostMapping", "/m/" + GeneratorUtil.toCamelCase(table.getTableName(), false), MediaType.APPLICATION_JSON_VALUE))
                .addModifiers(Modifier.PUBLIC)
                .addParameter(
                        ParameterSpec.builder(
                                //GeneratorUtil.className(PathEnum.ROOT.getPath(), PathEnum.DOMAIN_SUB.getPath(), table.getTableName())
                                GeneratorUtil.getClassName(entityClass)
                                , GeneratorUtil.toCamelCase(table.getTableName(), false)
                        )
                                .addAnnotation(AnnotationSpecEnum.getAnnotationSpec(AnnotationSpecEnum.REQUESTBODY))
                                .build()


                )
                .returns(ClassName.get(packageInfo.getDomain(), GeneratorUtil.toCamelCase(table.getTableName(), true)))
                .addStatement("return $L.insert($L)"
                        , GeneratorUtil.toCamelCase(GeneratorUtil.getClassName(repositoryClass).simpleName(), false)
                        , GeneratorUtil.toCamelCase(table.getTableName(), false))
                .build();

        MethodSpec getAllMethod = MethodSpec.methodBuilder("getAll" + GeneratorUtil.toCamelCase(table.getTableName(), true) + "s")
                .addAnnotation(GeneratorUtil.controllerMapping("GetMapping", "/m/" + GeneratorUtil.toCamelCase(table.getTableName(), false) + "s", ""))
                .addAnnotation(ResponseBody.class)
                .addModifiers(Modifier.PUBLIC)
                .addParameter(Pageable.class, "request")
                .returns(
                        ParameterizedTypeName.get(
                                ClassName.get("org.springframework.data.domain", "Page")
                                , GeneratorUtil.getClassName(entityClass)
                                //ClassName.get(PathEnum.DOMAIN.getPath(), GeneratorUtil.toCamelCase(table.getTableName(), true))
                        )
                )
                .addStatement("return $L.findAll(request)", GeneratorUtil.toCamelCase(GeneratorUtil.getClassName(repositoryClass).simpleName(), false))
                .build();

        TypeSpec.Builder controllerBuilder = TypeSpec.classBuilder(GeneratorUtil.toCamelCase(name, true) + "Controller")
                .addJavadoc("Automatically Generated Code! Do not Modify this file")
                .addAnnotation(AnnotationSpecEnum.getAnnotationSpec(AnnotationSpecEnum.RESTCONTROLLER))
                .addAnnotation(GeneratorUtil.controllerMapping("RequestMapping", "/api/v1", ""))
                .addModifiers(Modifier.PUBLIC)
                .addField(
                        FieldSpec.builder(
                                GeneratorUtil.getClassName(repositoryClass)
                                , GeneratorUtil.toCamelCase(GeneratorUtil.getClassName(repositoryClass).simpleName(), false)
                                , Modifier.PRIVATE
                        )
                                .addAnnotation(AnnotationSpecEnum.getAnnotationSpec(AnnotationSpecEnum.AUTOWIRED)).build()
                )
                .addMethod(addMethod)
                .addMethod(getAllMethod);

        return GeneratorUtil.javaFile(packageInfo.getController(), controllerBuilder.build());
    }

}