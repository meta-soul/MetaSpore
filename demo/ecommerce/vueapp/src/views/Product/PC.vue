<template>
  <div class="product-container" v-loading="!productData">
    <!-- <el-backtop target=".product-container" :visibility-height="200" :bottom="100" :right="130">Top</el-backtop> -->

    <ProductDesc
      :data="productData"
      v-if="productData"
      @enlarge="handleEnlarge"
    />
    <ListGroup
      :underline="false"
      acolor="blue"
      v-if="productData && isShow"
      :datas="listDatas"
    />
    <!-- <div v-if="!productData">正在加载</div> -->
    <!-- <el-backtop target=".page-component__scroll .el-scrollbar__wrap"></el-backtop> -->
  </div>
</template>
<script>
import ProductDesc from '@/components/PC/ProductDesc.vue';
import ListGroup from '@/components/PC/ListGroup.vue';
import { getOneProduct } from '@/api/index';
import { mapState } from 'vuex';
export default {
  components: {
    ProductDesc,
    ListGroup,
  },
  data() {
    return {
      // ListDatas: {
      //   title: '看了又看',
      //   items: [
      //     { img: FlowerUrl },
      //   ],
      // },
      isShow: true,
      productData: null,
      listDatas: {
        title: '看了又看',
      },
    };
  },
  computed: {
    ...mapState(['youlikeProducts']),
  },
  async created() {
    this.fetchProductData();

    // let result = await getOneProduct(this.$route.params.productId);
    // this.productData = result[0];
    // let res = await this.$store.dispatch('asyncGetYouLikeProducts');
    // this.listDatas = {
    //   title: '看了又看',
    //   items: res,
    // };
  },
  methods: {
    async fetchProductData() {
      // 滚动条回滚到顶部
      document.documentElement.scrollTo({
        top: 0,
      });
      this.productData = null; // 请求前先置为空
      let result = await getOneProduct(
        this.$store.state.cur_user,
        this.$route.params.productId
      );
      this.productData = result[0];
      this.listDatas.items = null; // 请求前先置为空
      let res;
      try {

         res = await this.$store.dispatch('asyncGetYouLikeProducts', {
          user_id: this.$store.state.cur_user,
          item_id: this.$route.params.productId,
        });
        this.listDatas = {
          title: '看了又看',
          items: res,
        };
      }catch(e) {
         this.listDatas = {
          title: '看了又看',
          items: [],
        };
      }
    },
    handleEnlarge(value) {
      this.isShow = value;
    },
  },
  watch: {
    $route() {
      this.fetchProductData();
    },
  },
};
</script>

<style scoped>
.product-container {
  height: calc(100vh - 46px);
  padding: 10px 15vw;
  box-sizing: border-box;
}
</style>
