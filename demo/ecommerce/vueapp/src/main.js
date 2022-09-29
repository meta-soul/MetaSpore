import Vue from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import VueLazyload from 'vue-lazyload';

Vue.use(ElementUI);

// 使用（以下两种方式可选）：
// 1.直接使用
// Vue.use(VueLazyload)
// 2.添加自定义选项
Vue.use(VueLazyload, {
  preLoad: 3,
  // error: require('@/assets/星星.webp'),
  // loading: require('@/assets/loading.svg'),
  attempt: 1,
  dispatchEvent: true, // 开启原生dom事件
});
Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');
