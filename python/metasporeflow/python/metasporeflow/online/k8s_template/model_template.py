default = {
    'name': "model-k8s-service",
    'port': 50000,
    'image': 'swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1',
    'consul_port': 8500,
    'consul_service': "consul-k8s-service",
    'watch_image': "swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/consul-watch-load:v1.0.0",
    'watch_port': 8080,
    'consul_key': "dev/",
    'endpoint_url': 'http://obs.cn-southwest-2.myhuaweicloud.com',
    'docker_secret': "regcred",
}
template = '''
apiVersion: v1
kind: Service
metadata:
  name: ${name}
  labels:
    app: model
spec:
  selector:
    app: model
  ports:
    - name: server
      port: ${port}
      targetPort: ${port}
  type: ClusterIP

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-configmap
data:
  config: |
    [default]
    s3 =
      addressing_style = virtual
      endpoint_url = ${endpoint_url}
    [plugins]
    endpoint = awscli_plugin_endpoint

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: model
    spec:
      imagePullSecrets:
      - name: ${docker_secret}
      containers:
      - name: model
        image: ${image}
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: ${port}
          name: server
        command: ["/opt/metaspore-serving/bin/metaspore-serving-bin", "-grpc_listen_port", "${port}", "-init_load_path", "/data/models"]
        volumeMounts:
        - name: data
          mountPath: /data/models
      - name: serving-watch-load
        image: "${watch_image}"
        imagePullPolicy: Always
        command: ["python", "consul_watch_load.py", "--model-root","/data/models", "--notify-port", "${port}", "--prefix", "${consul_key}"]
        ports:
          - name: http
            containerPort: ${watch_port}
            protocol: TCP
        volumeMounts:
          - name: aws-config-volume
            mountPath: /root/.aws
          - name: data
            mountPath: /data/models
        env:
          - name: AWS_ENDPOINT
            valueFrom:
              secretKeyRef:
                name: aws-secret
                key: aws_endpoint
                optional: true
          - name: AWS_ACCESS_KEY_ID
            valueFrom:
              secretKeyRef:
                name: aws-secret
                key: aws_access_key_id
                optional: true
          - name: AWS_SECRET_ACCESS_KEY
            valueFrom:
              secretKeyRef:
                name: aws-secret
                key: aws_secret_access_key
                optional: true
      - name: serving-curl-watch
        image: "${watch_image}"
        imagePullPolicy: Always
        command: ["/bin/sh","create_consul_watch.sh"]
        env:
          - name: CONSUL_IP
            value: ${consul_service}
          - name: POD_IP
            valueFrom:
              fieldRef:
                fieldPath: status.podIP
          - name: CONSUL_KEY_PREFIX
            value: "${consul_key}"
      volumes:
      - name: data
        emptyDir: {}
      - name: aws-config-volume
        configMap:
          name: aws-configmap
          items:
            - key: config
              path: config
'''
