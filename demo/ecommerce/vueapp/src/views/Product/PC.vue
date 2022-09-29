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
      //   title: 'You Also May Like',
      //   items: [
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //     { img: FlowerUrl },
      //     { img: Xing },
      //   ],
      // },
      isShow: true,
      productData: null,
      listDatas: {
        title: 'You Also May Like',
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
    //   title: 'You Also May Like',
    //   items: res,
    // };
  },
  methods: {
    async fetchProductData() {
      // 滚动条回滚到顶部
      document.documentElement.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
      let result = await getOneProduct(
        this.$store.state.cur_user,
        this.$route.params.productId
      );
      this.productData = result[0];
      this.listDatas.items = null; // 请求前先置为空
      let res = await this.$store.dispatch(
        'asyncGetYouLikeProducts',
        this.$store.state.cur_user
      );
      this.listDatas = {
        title: 'You Also May Like',
        items: res,
      };
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
