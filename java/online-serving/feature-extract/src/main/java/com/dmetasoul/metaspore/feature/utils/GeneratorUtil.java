package com.dmetasoul.metaspore.feature.utils;

import com.squareup.javapoet.*;
import com.dmetasoul.metaspore.feature.dao.Column;
import com.dmetasoul.metaspore.feature.javapoet.DataTypes;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.dmetasoul.metaspore.feature.javapoet.PackageInfo;
import org.apache.commons.text.CaseUtils;
public class GeneratorUtil {
    public static AnnotationSpec lombokAnnotationSpec(String simpleName) {
        return AnnotationSpec.builder(ClassName.get("lombok", simpleName)).build();
    }

    public static List<AnnotationSpec> entityAnnotations() {
        List<AnnotationSpec> annotationSpecList = new ArrayList<>();

        annotationSpecList.add(lombokAnnotationSpec("Data"));
        annotationSpecList.add(lombokAnnotationSpec("Builder"));
        annotationSpecList.add(lombokAnnotationSpec("AllArgsConstructor"));
        annotationSpecList.add(lombokAnnotationSpec("NoArgsConstructor"));

        return annotationSpecList;
    }

    public static FieldSpec entityField(Column column) {
        FieldSpec fieldSpec = null;
        try {
            fieldSpec = FieldSpec.builder(
                    Class.forName(DataTypes.getDataTypes().get(column.getColType()))
                    , CaseUtils.toCamelCase(column.getColName(), false, '_')
                    , Modifier.PRIVATE
            ).build();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return fieldSpec;
    }

    public static List<FieldSpec> entityFields(List<Column> columns) {
        List<FieldSpec> fieldSpecs = new ArrayList<>();

        columns.forEach(column -> {
            fieldSpecs.add(entityField(column));
        });

        return fieldSpecs;
    }


    public static ClassName getClassName(JavaFile javaFile) {
        return ClassName.get(javaFile.packageName, javaFile.typeSpec.name);
    }

    public static String toCamelCase(String name, Boolean firstUpper) {
        return firstUpper ? CaseUtils.toCamelCase(name, true, '_') : CaseUtils.toCamelCase(name, false, '_');
    }

    public static Class<?> className(String packageName, String subPackageName, String domainName) {
        Class<?> class1 = null;
        try {
            String className = String.format("%s.%s.%s", packageName, subPackageName, toCamelCase(domainName, true));
            class1 = Class.forName(className);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return class1;
    }

    public static AnnotationSpec controllerMapping(String simpleName, String path, String consumes) {
        AnnotationSpec.Builder builder = AnnotationSpec.builder(ClassName.get("org.springframework.web.bind.annotation", simpleName));
        if (path != null && (consumes == null || "".equals(consumes))) {
            builder.addMember("value", "path=$S", path);
        } else if (path != null) {
            builder.addMember("value", "path=$S, consumes = { $S }", path, consumes);
        }

        return builder.build();
    }

    public static JavaFile javaFile(String packageName, TypeSpec typeSpec) {

        JavaFile javaFile = JavaFile.builder(packageName, typeSpec)
                .skipJavaLangImports(true)
                .build();

        try {
            javaFile.writeTo(System.out);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return javaFile;
    }

    public static void writeJavaFile(JavaFile javaFile) {
        Path path = Paths.get(PackageInfo.PROJECT_ROOT_PATH);
        try {
            javaFile.writeTo(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeJavaFileToPath(JavaFile javaFile) {
        try {
            javaFile.writeTo(new File(PackageInfo.PROJECT_ROOT_PATH));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
