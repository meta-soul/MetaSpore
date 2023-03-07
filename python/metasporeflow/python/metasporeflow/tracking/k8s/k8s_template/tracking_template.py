default = {
  "port": 50001,
  "name": "tracking-k8s-service",
  "image": '132825542956.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dmetasoul-repo/tracking-service-test:latest',
  # "consul_port": 8500,
  # "consul_service": "consul-k8s-service",
  # "model_port": 50000,
  # "model_service": "model-k8s-service",
  # "mongo_port": 27017,
  # "mongo_service": "127.0.0.1",
  'domain': "tracking.dmetasoul.com",
  'namespace': "default",
}
template = '''
apiVersion: v1
kind: Service
metadata:
  name: ${name}
  namespace: ${namespace}
  labels:
    app: tracking
spec:
  selector:
    app: tracking
  ports:
    - name: http
      port: ${port}
      targetPort: ${port}
  type: ClusterIP
  
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tracking-ingress
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
  name: tracking
  namespace: ${namespace}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tracking
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: tracking
    spec:
      containers:
      - name: tracking
        image: ${image}
        imagePullPolicy: Always
        ports:
        - containerPort: ${port}
          name: http
        env:
        - name: SERVICE_PORT
          value: "${port}"
        command: ["python","entrypoint.py"]
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
