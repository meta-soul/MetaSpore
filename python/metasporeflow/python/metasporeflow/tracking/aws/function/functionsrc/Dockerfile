FROM 132825542956.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dmetasoul-repo/tracking-log-extension-image:latest AS layer
FROM public.ecr.aws/lambda/python:3.8
# Layer code
WORKDIR /opt
COPY --from=layer /opt/ .

# Function code
WORKDIR /var/task
COPY app.py .

CMD ["app.lambda_handler"]