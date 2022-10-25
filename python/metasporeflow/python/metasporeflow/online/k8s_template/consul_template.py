default = {
  'port': 8500,
  'name': "consul-k8s-service",
  'image': "consul:1.13.1",
  'domain': "huawei.dmetasoul.com",
}
template = '''
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: consul-ingress
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
apiVersion: v1
kind: Service
metadata:
  name: ${name}
  labels:
    app: consul
spec:
  selector:
    app: consul
  ports:
    - name: http
      port: 8500
      targetPort: ${port}
    - name: server
      port: 8300
      targetPort: 8300
      
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: consul-server-config
data:

---
apiVersion: policy/v1beta1
kind: PodDisruptionBudget
metadata:
  name: consul-server-budget
spec:
  selector:
    matchLabels:
      app: consul
  minAvailable: 2
  
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: consul-server
spec:
  serviceName: ${name}
  replicas: 3
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app: consul
  template:
    metadata:
      labels:
        app: consul
    spec:
      terminationGracePeriodSeconds: 10
      containers:
      - name: consul
        image: ${image}
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8500
          name: http
        - containerPort: 8300
          name: server
        env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        args:
        - "agent"
        - "-server"
        - "-advertise=$(POD_IP)"
        - "-bind=0.0.0.0"
        - "-bootstrap-expect=3"
        - "-datacenter=dc1"
        - "-config-dir=/consul/userconfig"
        - "-data-dir=/consul/data"
        - "-disable-host-node-id"
        - "-domain=cluster.local"
        - "-retry-join=consul-server-0.${name}.$(NAMESPACE).svc.cluster.local"
        - "-retry-join=consul-server-1.${name}.$(NAMESPACE).svc.cluster.local"
        - "-retry-join=consul-server-2.${name}.$(NAMESPACE).svc.cluster.local"
        - "-client=0.0.0.0"
        - "-ui"
        resources:
          limits:
            cpu: "100m"
            memory: "128Mi"
          requests:
            cpu: "100m"
            memory: "128Mi"
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - consul leave
        volumeMounts:
        - name: data
          mountPath: /consul/data
        - name: user-config
          mountPath: /consul/userconfig
      volumes:
      - name: user-config
        configMap:
          name: consul-server-config
      - name: data
        emptyDir: {}
      securityContext:
        fsGroup: 1000
'''
