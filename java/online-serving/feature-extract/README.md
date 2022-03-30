# Feature Extract

Feature Extract is a feature code automatic generation framework developed by [DMetaSoul](https://www.dmetasoul.com/). The user can customize the structure of the feature table by configuring the YAML file. The framework can generate the corresponding Spring Boot JPA Repository code file based on this file, including the relevant Query interface for accessing the database. Users can directly call this interface to access the database.



## Demo example

Full example: [Movielens Demo example link](../../../demo/movielens/online/README.md)



## Framework introduction

The framework is based on the spring boot project and is introduced through the maven plugin. Currently supports mongoDB, mysql.

- ### Table Definition

    Create a new item.yml in the resources/tables directory. tableName and collectionName represent the database name and table name respectively. dbType means select mongodb. The columns field represents the individual columns of the table.

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



- ### DB connection

  Take mongodb as an example, configure DB information under application.properties. Among them, spring.data.mongodb.field-naming-strategy=org.springframework.data.mapping.model.SnakeCaseFieldNamingStrategy refers to the underscore to camel case of the column name read from the DB.

    ```ini
    # mongodb
    spring.data.mongodb.host=localhost
    spring.data.mongodb.port=27017
    spring.data.mongodb.database="YOUR-DATABASE"
    spring.data.mongodb.username="YOUR-USERNAME"
    spring.data.mongodb.password="YOUR-PASSWORD"
    spring.jackson.default-property-inclusion=NON_NULL
    spring.data.mongodb.field-naming-strategy=org.springframework.data.mapping.model.SnakeCaseFieldNamingStrategy
    spring.jackson.serialization.indent_output=true
    ```



- ### project reference

1. Clone and install this plugin

   ```shell
   cd feature-extract
   mvn clean install
   ```
2. In the pom of your own spring boot project, refer to this plugin. The <packageName> configuration item fills in the package name where the springbootApplication (ie the entry class) is located
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
            <!-- Introduce feature-extract plugin -->
            <plugin>
                <groupId>com.dmetasoul.metaspore</groupId>
                <artifactId>feature-extract</artifactId>
                <version>1.0-SNAPSHOT</version>
                <configuration>
                    <packageName>"YOUR-SPRINGBOOTAPPLICATION-PACKAGE-NAME"</packageName>
                </configuration>
            </plugin>
            <!-- add generated-sources to root -->
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
   
   
4. Execute generate to verify whether the corresponding domain and repository are generated under target/generated-sources, and then perform the Test test
   ```shell
   // Generate the code in generated-sources
   mvn com.dmetasoul.metaspore:feature-extract:1.0-SNAPSHOT:generate
   ```

5. If it is developed in idea, you can mark the automatically generated code as Generated Sources Root, which is convenient for jumping and debugging

   Right click target/generated-sources/feature/java in idea, click "Mark Directory as" --> "Generated Sources Root".
   When the icon changes from red to blue, you can successfully jump to /debug/import/@Autowired