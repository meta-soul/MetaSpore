# Feature Extract

Feature Extract 是由 [DMetaSoul](https://www.dmetasoul.com/) 研发的特征代码自动生成框架。用户可以通过配置 YAML 文件来自定义特征表的结构，框架可以根据这个文件生成相应的 Spring Boot JPA Repository 代码文件，包含相关的访问数据库的 Query 接口，用户可以直接调用该接口进行数据库的访问。



## Demo 示例

完整示例：[Movielens Demo 示例链接](../../../../demo/movielens/online)



## 框架介绍

框架基于 spring boot 项目，通过 maven plugin 的方式引入。目前支持 mongoDB、mysql。

- ### Table 定义

resources/tables 目录下新建 item.yml。tableName、collectionName 分别代表数据库名和表名。dbType 表示选择 mongodb。columns 字段代表表的各个列。

```yaml
tableName: "item"
collectionName: "item"
dbType: "mongodb"
columns:
  - colName: "queryid"
    colType: "String"
  - colName: "movie_id"
    colType: "String"
  - colName: "title"
    colType: "String"
  - colName: "genre"
    colType: "String"
```



- ### DB 连接

  以 mongodb 为例，application.properties 下配置 DB 信息。其中 spring.data.mongodb.field-naming-strategy=org.springframework.data.mapping.model.SnakeCaseFieldNamingStrategy 是指从 DB 读出来的列名下划线转驼峰。

```
# mongodb
spring.data.mongodb.host=localhost
spring.data.mongodb.port=27017
spring.data.mongodb.database="YOUR-DATABASE"
spring.data.mongodb.username="YOUR-USERNAME"
spring.data.mongodb.password="YOUR-PASSWORD"
spring.jackson.default-property-inclusion=NON_NULL
spring.data.mongodb.field-naming-strategy=org.springframework.data.mapping.model.SnakeCaseFieldNamingStrategy
spring.jackson.serialization.indent_output = true
```



- ### 项目引用

1. 克隆、安装本插件

   ```shell
   cd feature-extract
   mvn clean install
   ```
2. 自己的 spring boot 项目的 pom 中, 引用本插件. 其中 <packageName> 配置项填写 springbootApplication(即入口类) 所在包名
   ```xml
   <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.8.1</version>
                <configuration>
                    <source>1.8</source>
                    <target>1.8</target>
                </configuration>
            </plugin>
            <!-- 引入 feature-extract 插件 -->
            <plugin>
                <groupId>com.dmetasoul</groupId>
                <artifactId>feature-extract</artifactId>
                <version>1.0-SNAPSHOT</version>
                <configuration>
                    <packageName>"YOUR-SPRINGBOOTAPPLICATION-PACKAGE-NAME"</packageName>
                </configuration>
            </plugin>
            <!-- 将 generated-sources 加入 root -->
            <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>build-helper-maven-plugin</artifactId>
                <version>3.2.0</version>
                <executions>
                    <execution>
                        <phase>generate-sources</phase>
                        <goals>
                            <goal>add-source</goal>
                        </goals>
                        <configuration>
                            <sources>
                                <source>${project.build.directory}/generated-sources/feature/java</source>
                            </sources>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
   </build>
   ```
   
   
4. 执行 generate ，验证target/generated-sources 下是否产出相应 domain 和 repository, 然后进行 Test 测试
   ```shell
   // 生成 generated-sources 里面的代码
   mvn com.dmetasoul:feature-extract:1.0-SNAPSHOT:generate
   
   ```

5. 如果是 idea 中开发, 可以把自动生成的代码标记为 Generated Sources Root, 方便跳转和调试 
   ```shell
   idea 中右键 target/generated-sources/feature/java, 点击 "Mark Directory as" --> "Generated Sources Root".
   当图标由红色变成蓝色,即可顺利 跳转/debug/import/@Autowired
   ```