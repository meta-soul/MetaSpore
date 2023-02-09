# Build Tracking Image

## aws extension
```bash
cd extension
docker build -t YOUR_REPO/tracking-log-extension-image:latest .
docker tag YOUR_REPO/tracking-log-extension-image:latest YOUR_AWS_ECR/tracking-log-extension-image:latest
aws ecr create-repository --repository-name YOUR_AWS_ECR/tracking-log-extension-image --image-scanning-configuration scanOnPush=true --region YOUR_AWS_REGION
docker push YOUR_AWS_ECR/tracking-log-extension-image:latest
```

## aws function
```bash
cd function/functionsrc
docker build -t YOUR_REPO/tracking-log-extension-function:latest .
docker tag YOUR_REPO/tracking-log-extension-function:latest YOUR_AWS_ECR/tracking-log-extension-function:latest
aws ecr create-repository --repository-name YOUR_REPO/tracking-log-extension-function --image-scanning-configuration scanOnPush=true --region YOUR_AWS_REGION
docker push YOUR_AWS_ECR/tracking-log-extension-function:latest
```

## curl
```bash
curl -X POST https://YOUR_AWS_API_GATEWAY_ENDPOINT -d '{"payload": "hello"}'
```