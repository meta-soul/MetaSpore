# SERVING CHART

Deployment of Serving service on Kubernetes using helm

<br/>

## Instructions

1. Define environment variables
   ```
   export IMAGE_REPOSITORY=$YOUR-REPOSITORY/metaspore-serving-release
   export IMAGE_TAG=YOUR-TAG
   export NAMESPACE=YOUR-NAMESPACE
   export STORAGECLASS=YOUR-STORAGECLASS
   ```
2. Execute the installation command to deploy the Serving service
   ```
   helm install serving -n $NAMESPACE ./chart \
     --set image.repository=$IMAGE_REPOSITORY \
     --set image.tag=$IMAGE_TAG \
     --set storageclass=$STORAGECLASS
   ```