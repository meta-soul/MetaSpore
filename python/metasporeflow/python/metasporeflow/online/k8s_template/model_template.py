keys = [
    'name',
    'port',
    'node_port',
    'image',
    'consul_port',
    'watch_image',
    'consul_key',
]
template = '''
apiVersion: v1
kind: Service
metadata:
  name: model-serving
  labels:
    app: model
spec:
  selector:
    app: model
  ports:
    - name: server
      port: ${port}
      targetPort: ${port}
      nodePort: ${node_port}
  type: NodePort
  
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
      containers:
      - name: ${container_name}
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
            value: consul-ui
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
