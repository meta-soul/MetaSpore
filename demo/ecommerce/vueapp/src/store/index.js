import Vue from 'vue';
import Vuex from 'vuex';
import {
  getAllProducts,
  getOneProduct,
  LookAndLookProducts
} from '@/api/index';

Vue.use(Vuex);

if (!localStorage.getItem('user')) {
  localStorage.setItem('user', JSON.stringify('A23P7HJBRQ0F7L')); // 初始化
}

export default new Vuex.Store({
  state: {
    mainAllProducts: [],
    youlikeProducts: [],
    cur_user: JSON.parse(localStorage.getItem('user')),
    // cur_user: "A14EI4NEAWCH18"
  },
  getters: {},
  mutations: {
    setMainAllProducts(state, payload) {
      state.mainAllProducts = payload;
    },
    setYoulikeProducts(state, payload) {
      state.mainAllProducts = payload;
    },
    setCurUser(state, payload) {
      localStorage.setItem('user', JSON.stringify(payload));
      state.cur_user = payload;
    },
  },
  actions: {
    async asyncGetAllProducts({ commit }, user_id) {
      // 获取首页所有数据
      let result = await getAllProducts(user_id);
      // result = result.map((item) => ({
      //   user_id: item.user_id,
      //   item_id: item.item_id,
      // }));
      // let arr = [];
      // // 每一项获取单个数据的详情页
      // for (let i = 0; i < result.length && i < 10; i++) {
      //   arr.push(getOneProduct(user_id, result[i].item_id));
      // }
      // result = await Promise.all(arr);
      // result = result.map((item) => item[0]);
      commit('setMainAllProducts', result);
      return result;
    },
    async asyncGetYouLikeProducts({ commit }, {user_id, item_id}) {
      let result = await LookAndLookProducts(user_id, item_id);
      commit('setYoulikeProducts', result);
      return result;
    },
  },
  modules: {},
});
