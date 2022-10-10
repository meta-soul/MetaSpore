const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      "/service": {
        target: "http://127.0.0.1:13013"
      }
    }
  }
})
