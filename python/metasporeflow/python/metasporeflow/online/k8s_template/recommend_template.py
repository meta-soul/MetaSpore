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
}
template = '''
apiVersion: v1
kind: Service
metadata:
  name: ${name}
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
spec:
  rules:
  - host: ${name}.${domain}
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
        imagePullPolicy: IfNotPresent
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
        resources:
          limits:
            cpu: "2000m"
            memory: "3072Mi"
          requests:
            cpu: "1000m"
            memory: "1280Mi"
'''
