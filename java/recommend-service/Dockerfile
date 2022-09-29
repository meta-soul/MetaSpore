FROM openjdk:11
EXPOSE 9081
ARG JAR_FILE=target/recommend-service-1.0-SNAPSHOT.jar
ADD ${JAR_FILE} recommend-service-1.0-SNAPSHOT.jar
CMD ["java"," ","-jar","recommend-service-1.0-SNAPSHOT.jar"]
