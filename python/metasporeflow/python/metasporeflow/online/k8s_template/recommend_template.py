default = {
  "port": 13013,
  "name": "recommend-k8s-service",
  "image": 'swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service:1.0.0',
  "consul_port": 8500,
  "consul_service": "consul-k8s-service",
  "model_port": 50000,
  "model_service": "model-k8s-service",
  "mongo_port": 27017,
  "mongo_service": "127.0.0.1",
  'domain': "huawei.dmetasoul.com",
  'namespace': "default",
}
template = '''
apiVersion: v1
kind: Service
metadata:
  name: ${name}
  namespace: ${namespace}
  labels:
    app: recommend
spec:
  selector:
    app: recommend
  ports:
    - name: http
      port: ${port}
      targetPort: ${port}
  type: ClusterIP
  
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: recommend-ingress
  namespace: ${namespace}
spec:
  rules:
  - host: ${name}-${namespace}.${domain}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ${name}
            port:
              number: ${port}
              
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommend
  namespace: ${namespace}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommend
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: recommend
    spec:
      containers:
      - name: recommend
        image: ${image}
        imagePullPolicy: Always
        ports:
        - containerPort: ${port}
          name: http
        env:
        - name: CONSUL_HOST
          value: ${consul_service}
        - name: CONSUL_PORT
          value: "${consul_port}"
        - name: MODEL_HOST
          value: ${model_service}
        - name: MODEL_PORT
          value: "${model_port}"
        - name: MONGO_HOST
          value: "${mongo_service}"
        - name: MONGO_PORT
          value: "${mongo_port}"
        - name: SERVICE_PORT
          value: "${port}"
        command: ["java", "-Xmx2048M", "-Xms2048M", "-Xmn768M", "-XX:MaxMetaspaceSize=256M", "-XX:MetaspaceSize=256M", "-jar", "recommend-service-1.0-SNAPSHOT.jar"]
        readinessProbe:
          httpGet:
            path: /actuator/pullConfig
            port: ${port}
          initialDelaySeconds: 10
          periodSeconds: 60
        resources:
          limits:
            cpu: "2000m"
            memory: "3072Mi"
          requests:
            cpu: "1000m"
            memory: "1280Mi"
'''
