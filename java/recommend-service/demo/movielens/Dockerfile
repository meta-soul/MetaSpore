FROM openjdk:11
EXPOSE 8080
ARG JAR_FILE=../../target/recommend-service-1.0-SNAPSHOT.jar
ADD ${JAR_FILE} recommend-service-1.0-SNAPSHOT.jar
CMD ["java"," ","-jar","recommend-service-1.0-SNAPSHOT.jar"]