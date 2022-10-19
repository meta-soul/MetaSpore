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

package com.dmetasoul.metaspore.feature.utils;

import com.github.javaparser.ast.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.nodeTypes.NodeWithName;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.Type;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.dmetasoul.metaspore.feature.dao.TableAttributes;
import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.github.javaparser.ast.Modifier.Keyword.PUBLIC;
import static com.google.gson.JsonParser.parseString;

public class ParserHelper {
    public static Optional<FieldDeclaration> findFieldByAnnotation(TypeDeclaration<?> ty, String name) {
        for (FieldDeclaration f : ty.getFields()) {
            for (AnnotationExpr an : f.getAnnotations()) {
                if (an.getNameAsString().equals(name)) {
                    return Optional.of(f);
                }
            }
        }
        return Optional.empty();
    }

    public static Optional<TypeDeclaration<?>> findTypeByAnnotation(CompilationUnit parse, String annotation) {
        for (TypeDeclaration<?> ty : parse.getTypes()) {
            String clazz = ty.getAnnotationByName(annotation).map(NodeWithName::getNameAsString).orElse(null);
            if (clazz != null) {
                return Optional.of(ty);
            }
        }
        return Optional.empty();
    }

    public static Optional<Map.Entry<ClassOrInterfaceDeclaration, ClassOrInterfaceType>> findTypeBySuperclass(CompilationUnit parse, String classOrInterface) {
        for (TypeDeclaration<?> ty : parse.getTypes()) {
            if (ty instanceof ClassOrInterfaceDeclaration) {
                ClassOrInterfaceDeclaration type = (ClassOrInterfaceDeclaration) ty;
                for (ClassOrInterfaceType clazz : type.getExtendedTypes()) {
                    if (clazz.getNameAsString().equals(classOrInterface)) {
                        return Optional.of(new AbstractMap.SimpleEntry<>(type, clazz));
                    }
                }
                for (ClassOrInterfaceType clazz : type.getImplementedTypes()) {
                    if (clazz.getNameAsString().equals(classOrInterface)) {
                        return Optional.of(new AbstractMap.SimpleEntry<>(type, clazz));
                    }
                }
            }
        }
        return Optional.empty();
    }

    public static List<Optional<ImportDeclaration>> findImportClass(CompilationUnit parse, Type... type) {
        return findImportClass(parse, Arrays.stream(type));
    }

    public static List<Optional<ImportDeclaration>> findImportClass(CompilationUnit parse, Collection<Parameter> parameters) {
        return findImportClass(parse, parameters.stream().map(Parameter::getType));
    }

    private static List<Optional<ImportDeclaration>> findImportClass(CompilationUnit parse, Stream<Type> type) {
        List<Optional<ImportDeclaration>> list = new CopyOnWriteArrayList<>();
        type.forEach(ty -> {
            List<Optional<ImportDeclaration>> os = findImportClass(parse, ty);
            list.addAll(os);
        });
        return list;
    }

    public static List<Optional<ImportDeclaration>> findImportClass(CompilationUnit parse, Type type) {
        return findImportClass(parse, type, (File) null);
    }

    public static List<Optional<ImportDeclaration>> findImportClass(CompilationUnit parse, Type type, File file) {
        List<Optional<ImportDeclaration>> list = new CopyOnWriteArrayList<>();
        if (type instanceof ClassOrInterfaceType) {
            ClassOrInterfaceType ci = (ClassOrInterfaceType) type;
            Optional<ImportDeclaration> cs = findImportClass(parse, ci.getNameAsString());
            if (cs.isPresent()) {
            } else if (file != null) {
                File root = tryRootByPackage(file, parse);
                cs = findImportClassFromClassPath(root, parse, type);
            }
            list.add(cs);
            ci.getTypeArguments().ifPresent(oo -> {
                oo.stream().forEach(e -> {
                    List<Optional<ImportDeclaration>> os = findImportClass(parse, e, file); // 递归
                    list.addAll(os);
                });
            });
        }
        return list;
    }

    public static File tryRootByPackage(File file, CompilationUnit parse) {
        File root = file;
        {
            String[] pck = parse.getPackageDeclaration().map(e -> e.getNameAsString()).orElse("").split(Pattern.quote("."));
            for (int i = pck.length; ; i--) {
                root = root.getParentFile();
                if (i <= 0 || !root.getName().equals(pck[i - 1])) {
                    break;
                }
            }
        }
        return root;
    }

    public static MethodDeclaration delegate(FieldDeclaration field, MethodDeclaration method) {
        ClassOrInterfaceDeclaration toClazz = (ClassOrInterfaceDeclaration) field.getParentNode().get();
        FieldAccessExpr expr = new FieldAccessExpr(new ThisExpr(), field.getVariable(0).getNameAsString());
        return delegate(toClazz, expr, method);
    }

    public static MethodDeclaration delegate(ClassOrInterfaceDeclaration toClazz, FieldAccessExpr fieldAccessExpr, MethodDeclaration method) {
        MethodDeclaration md = toClazz.addMethod(method.getNameAsString(), PUBLIC);
        md.setParameters(method.getParameters());
        md.setType(method.getType());
        CompilationUnit from = getRootNode(method).get();
        CompilationUnit target = getRootNode(toClazz).get();

        ParserHelper.findImportClass(from, md.getType()).stream().filter(Optional::isPresent).map(Optional::get).forEach(target::addImport);
        ParserHelper.findImportClass(from, method.getParameters()).stream().filter(Optional::isPresent).map(Optional::get).forEach(target::addImport);

        BlockStmt stmt = md.getBody().get();
        List<NameExpr> collect = method.getParameters().stream().map(m -> new NameExpr(m.getNameAsString())).collect(Collectors.toList());
        NodeList<Expression> list = new NodeList<>();
        list.addAll(collect);
        stmt.addStatement(new ReturnStmt(new MethodCallExpr(fieldAccessExpr, method.getNameAsString(), list)));
        return md;
    }

    public static Optional<CompilationUnit> getRootNode(Node node) {
        if (node instanceof CompilationUnit) {
            return Optional.of((CompilationUnit) node);
        } else {
            Optional<Node> p = node.getParentNode();
            if (p.isPresent()) {
                return getRootNode(p.get());
            }
        }
        return Optional.empty();
    }

    public static Optional<ImportDeclaration> findImportClassFromClassPath(File javaFile, CompilationUnit parse, Type type) {
        if (type instanceof ClassOrInterfaceType) {
            ClassOrInterfaceType tp = (ClassOrInterfaceType) type;
            return parse.getPackageDeclaration().flatMap(e -> {
                String prt = e.getNameAsString().replace('.', File.separatorChar);
                File file = new File(new File(javaFile, prt), tp.getNameAsString() + ".java");
                if (file.exists()) {
                    return Optional.of(new ImportDeclaration(e.getNameAsString() + "." + tp.getNameAsString(), false, false));
                }
                return Optional.empty();
            });
        }
        return Optional.empty();
    }

    public static Optional<ImportDeclaration> findImportClass(CompilationUnit parse, String classOrInterface) {
        for (ImportDeclaration impt : parse.getImports()) {
            String path = impt.getNameAsString();
            if (path.endsWith("." + classOrInterface)) {
                return Optional.of(impt);
            } else {
                if (impt.isAsterisk()) {
                    for (char c : new char[]{'.', '$'}) {
                        try {
                            Class<?> cz = Class.forName(path + c + classOrInterface);
                            return Optional.of(new ImportDeclaration(cz.getName(), false, false));
                        } catch (ClassNotFoundException e) {
                        }
                    }
                }
            }
        }
        return Optional.empty();
    }

    public static ImportDeclaration findImportClass(ClassOrInterfaceDeclaration clazz) {
        List<String> list = new ArrayList();
        find(clazz, list).ifPresent(c ->
                c.getPackageDeclaration().ifPresent(p -> {
                    list.add(0, p.getNameAsString());
                }));
        String join = list.stream().collect(Collectors.joining("."));
        return new ImportDeclaration(join, false, false);
    }

    private static Optional<CompilationUnit> find(Node node, List<String> append) {
        if (node instanceof ClassOrInterfaceDeclaration) {
            append.add(((ClassOrInterfaceDeclaration) node).getNameAsString());
        }
        Optional<Node> parent = node.getParentNode();
        if (parent.isPresent()) {
            return find(parent.get(), append);
        }
        if (node instanceof CompilationUnit) {
            return Optional.of((CompilationUnit) node);
        }
        return Optional.empty();
    }

    public static String firstLower(String fieldName) {
        return fieldName.substring(0, 1).toLowerCase() + fieldName.substring(1);
    }

    public static String firstUpper(String fieldName) {
        return fieldName.substring(0, 1).toUpperCase() + fieldName.substring(1);
    }

    public static TableAttributes getTableFromYaml(String yamlPath) {

        File f = new File(yamlPath);
        Yaml yaml = new Yaml();
        Gson gson = null;
        String json = null;
        try {
            Object load = yaml.load(new FileReader(f));
            gson = new GsonBuilder().setPrettyPrinting().create();
            json = gson.toJson(load, LinkedHashMap.class);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        JsonObject jsonObject = parseString(json).getAsJsonObject();
        TableAttributes table = gson.fromJson(jsonObject, TableAttributes.class);
        return table;
    }
}