const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      "/service": {
        target: "http://recommend-demo.dmetasoul.com"
      }
    },
  },
})
