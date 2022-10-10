# Ecommerce Recommendation SaaS Demo

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).

### Build Docker Image
```
docker build -t swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/ecommerce-vue-app -f dockerfile .
```

### Start Docker
```
docker-compose up -d
```

### Test Web APP
Connect the server through ssh tunnel and open `http://localhost:41730`

### Shutdown Docker
```
docker-compose down
```