default = {
    "port": 50001,
    "name": "tracking-k8s-service",
    "image": 'swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/tracking-service-test:v1.0.4',
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
  replicas: 1
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
        - name: PORT
          value: "${port}"
        - name: UPLOAD_TYPE
          value: "${uploadType}"
        - name: UPLOAD_PATH
          value: "${uploadPath}"
        - name: ACCESS_KEY_ID
          value: "${accessKeyId}"
        - name: SECRET_ACCESS_KEY
          value: "${secretAccessKey}"
        - name: ENDPOINT
          value: "${endpoint}"
        - name: UPLOAD_WHEN
          value: "${uploadWhen}"
        - name: UPLOAD_INTERVAL
          value: "${uploadInterval}"
        - name: UPLOAD_BACKUP_COUNT
          value: "${uploadBackupCount}"
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
