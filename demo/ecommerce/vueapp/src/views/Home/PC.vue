<template>
  <div class="home-container" v-loading="!datas">
    <template v-if="datas">
      <div class="user-list">
        <el-dropdown split-button type="primary" @command="handleCommand">
          {{ $store.state.cur_user }}
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item command="A14EI4NEAWCH18"
              >A14EI4NEAWCH18</el-dropdown-item
            >
            <el-dropdown-item command="A1P62PK6QVH8LV"
              >A1P62PK6QVH8LV</el-dropdown-item
            >
            <el-dropdown-item command="A1GJTTKWXX3W5B"
              >A1GJTTKWXX3W5B</el-dropdown-item
            >
            <el-dropdown-item command="A1COZP39EL13YZ"
              >A1COZP39EL13YZ</el-dropdown-item
            >
          </el-dropdown-menu>
        </el-dropdown>
      </div>
      <list-group :underline="true" :datas="datas"></list-group>
    </template>
  </div>
</template>

<script>
// @ is an alias to /src
import ListGroup from '@/components/PC/ListGroup.vue';
export default {
  components: {
    ListGroup,
  },
  data() {
    return {
      datas: null,
      user: this.$store.state.cur_user,
    };
  },
  async created() {
    let result = await this.$store.dispatch(
      'asyncGetAllProducts',
      this.$store.state.cur_user
    );
    this.datas = {
      title: 'Featured Products',
      items: result,
    };
  },
  methods: {
    async handleCommand(command) {
      if (command === this.$store.state.cur_user) {
        return;
      }
      this.$store.commit('setCurUser', command);
      this.datas = null;
      let result = await this.$store.dispatch('asyncGetAllProducts', command);
      this.datas = {
        title: 'Featured Products',
        items: result,
      };
    },
  },
};
</script>
<style scoped lang="less">
.home-container {
  height: calc(100vh - 46px);
  padding: 5px 100px;
  box-sizing: border-box;
  .user-list {
    text-align: end;
  }
}
</style>
