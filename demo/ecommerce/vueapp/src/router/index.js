import Vue from 'vue';
import VueRouter from 'vue-router';
import { isMobile } from '@/utils/isMobile';
// import HomeView from '../views/HomeView.vue';

Vue.use(VueRouter);


//默认路由
export const routes = [
  {
    path: '/',
    redirect: '/home',
  },
];

//pc端的路由
export const pcRoutes = [
  // {
  //   path: '/',
  //   redirect: {
  //     name: "user",
  //     params: {
  //       userId: "A14EI4NEAWCH18"
  //     }
  //   },
  // },
  {
    path: '/',
    name: 'user',
    component: () =>
      import(/* webpackChunkName: "about" */ '../views/Home/PC.vue'),
  },
  {
    path: '/product/:productId',
    name: 'product',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ '../views/Product/PC.vue'),
  },
];
//移动端端的路由
export const mobileRoutes = [
  {
    path: '/',
    name: 'home',
    component: () =>
      import(/* webpackChunkName: "about" */ '../views/Home/Mobile.vue'),
  },
  {
    path: '/product/:productId',
    name: 'product',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ '../views/Product/Mobile.vue'),
  },
];

const router = new VueRouter({
  routes: isMobile() ? mobileRoutes : pcRoutes,
  mode: 'history',
});

// export default router;
// const createRouter = () =>
//   new VueRouter({
//     scrollBehavior: () => ({ y: 0 }),
//     mode: "history",
//     routes: routes,
//   });
 
// const router = createRouter();
 
// // Detail see: https://github.com/vuejs/vue-router/issues/1234#issuecomment-357941465
// export function resetRouter() {
//   const newRouter = createRouter();
//   router.matcher = newRouter.matcher; // reset router
// }
 
export default router;
