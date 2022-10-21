keys = [
  "port",
  "node_port",
  "name",
  "image",
  "consul_port",
]
template = '''
apiVersion: v1
kind: Service
metadata:
  name: recommend-service
  labels:
    app: recommend
spec:
  selector:
    app: recommend
  ports:
    - name: http
      port: ${port}
      targetPort: ${port}
      nodePort: ${node_port}
  type: NodePort

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommend-service
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
      - name: ${name}
        image: ${image}
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: ${port}
          name: http
        env:
        - name: CONSUL_HOST
          value: consul-ui
        - name: CONSUL_PORT
          value: "${consul_port}"
        command: ["java", "-Xmx2048M", "-Xms2048M", "-Xmn768M", "-XX:MaxMetaspaceSize=256M", "-XX:MetaspaceSize=256M", "-jar", "recommend-service-1.0-SNAPSHOT.jar"]
        resources:
          limits:
            cpu: "2000m"
            memory: "3072Mi"
          requests:
            cpu: "1000m"
            memory: "1280Mi"
'''
