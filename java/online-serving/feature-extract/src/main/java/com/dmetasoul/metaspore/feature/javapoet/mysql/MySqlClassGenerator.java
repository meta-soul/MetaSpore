package com.dmetasoul.metaspore.feature.javapoet.mysql;

import com.squareup.javapoet.*;
import com.dmetasoul.metaspore.feature.dao.TableAttributes;
import com.dmetasoul.metaspore.feature.javapoet.BaseClassGenerator;
import com.dmetasoul.metaspore.feature.javapoet.PackageInfo;
import com.dmetasoul.metaspore.feature.utils.GeneratorUtil;
import org.apache.commons.text.CaseUtils;

import javax.lang.model.element.Modifier;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import java.util.Collection;
import java.util.Optional;

public class MySqlClassGenerator implements BaseClassGenerator {

    public static AnnotationSpec createJpaEntityAnnotationSpec() {
        return AnnotationSpec.builder(ClassName.get("javax.persistence", "Entity")).build();
    }

    /**
     * 创建POJO
     *
     * @param tableAttributes
     * @param packageInfo
     * @return
     */
    @Override
    public JavaFile generateDomain(TableAttributes tableAttributes, PackageInfo packageInfo) {
        AnnotationSpec annotationSpec = AnnotationSpec.builder(GeneratedValue.class)
                .addMember("strategy",
                        CodeBlock.builder()
                                .add("$T.AUTO", ClassName.bestGuess("javax.persistence.GenerationType"))
                                .build()
                ).build();

        FieldSpec idField = FieldSpec.builder(ClassName.get(Long.class), "Id", Modifier.PRIVATE)
                .addAnnotation(Id.class)
                .addAnnotation(annotationSpec)
                .build();

        TypeSpec typeSpec = TypeSpec.classBuilder(CaseUtils.toCamelCase(tableAttributes.getTableName(), true, '_'))
                .addJavadoc("Automatically Generated Code! Do not Modify this file")
                .addModifiers(Modifier.PUBLIC)
                .addAnnotation(createJpaEntityAnnotationSpec())
                .addAnnotations(GeneratorUtil.entityAnnotations())
                .addField(idField)
                .addFields(GeneratorUtil.entityFields(tableAttributes.getColumns()))
                .build();

        return GeneratorUtil.javaFile(packageInfo.getDomain(), typeSpec);
    }

    @Override
    public JavaFile generateRepository(String repoPrefixName, PackageInfo packageInfo,
                                       JavaFile daoClass, TableAttributes table) {
        ParameterizedTypeName parameterizedTypeName = ParameterizedTypeName.get(
                ClassName.bestGuess("org.springframework.data.jpa.repository.JpaRepository")
                , GeneratorUtil.getClassName(daoClass)
                , ClassName.get(Long.class)
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

    /**
     * 根据POJO与Repository生成controller
     *
     * @param name
     * @param packageInfo
     * @param entityClass
     * @param table
     * @return
     */

//    public static JavaFile generateController(String name, PackageInfo packageInfo, JavaFile entityClass, JavaFile repositoryClass, TableAttributes table) {
//        MethodSpec addMethod = MethodSpec.methodBuilder("addNew" + GeneratorUtil.toCamelCase(table.getTableName(), true))
//                .addAnnotation(GeneratorUtil.controllerMapping("PostMapping", GeneratorUtil.toCamelCase(table.getTableName(), false), MediaType.APPLICATION_JSON_VALUE))
//                .addModifiers(Modifier.PUBLIC)
//                .addParameter(
//                        ParameterSpec.builder(
//                                        //GeneratorUtil.className(PathEnum.ROOT.getPath(), PathEnum.DOMAIN_SUB.getPath(), table.getTableName())
//                                        GeneratorUtil.getClassName(entityClass)
//                                        , GeneratorUtil.toCamelCase(table.getTableName(), false)
//                                )
//                                .addAnnotation(AnnotationSpecEnum.getAnnotationSpec(AnnotationSpecEnum.REQUESTBODY))
//                                .build()
//
//
//                )
//                .returns(ClassName.get(packageInfo.getDomain(), GeneratorUtil.toCamelCase(table.getTableName(), true)))
//                .addStatement("return $L.save($L)"
//                        , GeneratorUtil.toCamelCase(GeneratorUtil.getClassName(repositoryClass).simpleName(), false)
//                        , GeneratorUtil.toCamelCase(table.getTableName(), false))
//                .build();
//
//        MethodSpec getALLMethod = MethodSpec.methodBuilder("getAll" + GeneratorUtil.toCamelCase(table.getTableName(), true) + "s")
//                .addAnnotation(GeneratorUtil.controllerMapping("GetMapping", GeneratorUtil.toCamelCase(table.getTableName(), false) + "s", ""))
//                .addAnnotation(ResponseBody.class)
//                .addModifiers(Modifier.PUBLIC)
//                .addParameter(Pageable.class, "request")
//                .returns(
//                        ParameterizedTypeName.get(
//                                ClassName.get("org.springframework.data.domain", "Page")
//                                , GeneratorUtil.getClassName(entityClass)
//                                //ClassName.get(PathEnum.DOMAIN.getPath(), GeneratorUtil.toCamelCase(table.getTableName(), true))
//                        )
//                )
//                .addStatement("return $L.findAll(request)", GeneratorUtil.toCamelCase(GeneratorUtil.getClassName(repositoryClass).simpleName(), false))
//                .build();
//
//        TypeSpec.Builder controllerBuilder = TypeSpec.classBuilder(name + "Controller")
//                .addJavadoc("Automatically Generated Code! Do not Modify this file")
//                .addAnnotation(AnnotationSpecEnum.getAnnotationSpec(AnnotationSpecEnum.RESTCONTROLLER))
//                .addAnnotation(GeneratorUtil.controllerMapping("RequestMapping", "/api/v1", ""))
//                .addModifiers(Modifier.PUBLIC)
//                .addField(
//                        FieldSpec.builder(
//                                        GeneratorUtil.getClassName(repositoryClass)
//                                        , GeneratorUtil.toCamelCase(GeneratorUtil.getClassName(repositoryClass).simpleName(), false)
//                                        , Modifier.PRIVATE
//                                )
//                                .addAnnotation(AnnotationSpecEnum.getAnnotationSpec(AnnotationSpecEnum.AUTOWIRED))
//                                .build()
//                )
//                .addMethod(addMethod)
//                .addMethod(getALLMethod);
//
//        return GeneratorUtil.javaFile(packageInfo.getController(), controllerBuilder.build());
//    }

}
