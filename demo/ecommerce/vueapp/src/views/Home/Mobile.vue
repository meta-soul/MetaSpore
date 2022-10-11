<template>
  <div class="home-container" v-loading="!datas">
    <template v-if="datas">
      <div class="user-list">
        <el-dropdown @command="handleCommand">
          <div class="avatar-container">
            <el-avatar
              src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"
            ></el-avatar>
            <span class="user">{{ $store.state.cur_user }}</span>
          </div>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item
              v-for="item in getUserList"
              :key="item"
              :command="item"
              >
             {{ item }}
              </el-dropdown-item
            >
            <!-- <el-dropdown-item command="A120RH58WVY4W6"
              >A120RH58WVY4W6</el-dropdown-item
            >
            <el-dropdown-item command="A3FMK5TW8HVBZZ"
              >A3FMK5TW8HVBZZ</el-dropdown-item
            >
            <el-dropdown-item command="A2RM4WAQNE0PQN"
              >A2RM4WAQNE0PQN</el-dropdown-item
            >
            <el-dropdown-item command="ANOYMOUS"
              >ANOYMOUS</el-dropdown-item
            > -->
          </el-dropdown-menu>
        </el-dropdown>
      </div>
      <list-group :underline="true" :datas="datas"></list-group>
    </template>
    <!-- <list-group :underline="true" v-if="datas" :datas="datas"></list-group> -->
  </div>
</template>

<script>
// @ is an alias to /src
import ListGroup from '@/components/Mobile/ListGroup.vue';
export default {
  components: {
    ListGroup,
  },
  computed: {
    getUserList() {
      return this.userList.filter(
        (item) => item !== this.$store.state.cur_user
      );
    },
  },
  data() {
    return {
      datas: null,
      user: this.$store.state.cur_user,
      userList: [
        "A23P7HJBRQ0F7L",
        "A1EL1KCQTW6P32",
        "A2E9076GV6LE6F",
        // 'A14EI4NEAWCH18',
        // 'A120RH58WVY4W6',
        // 'A3FMK5TW8HVBZZ',
        // 'A2RM4WAQNE0PQN',
        'ANOYMOUS',
      ],
    };
  },
  async created() {
    let result = await this.$store.dispatch(
      'asyncGetAllProducts',
      this.$store.state.cur_user
    );
    this.datas = {
      title: '猜你喜欢',
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
        title: '猜你喜欢',
        items: result,
      };
      this.isCollapse = false;
    },
  },
};
</script>
<style scoped lang="less">
@import url('//at.alicdn.com/t/c/font_3668378_h7zlfk3oz0h.css');
.home-container {
  height: calc(100vh - 46px);
  padding: 5px 0.5rem;
  box-sizing: border-box;
}
.user-list {
  text-align: end;
}
.avatar-container {
    cursor: pointer;
    margin-top: .3rem;
    display: flex;
    align-items: center;
  }
  .user {
    margin-left: .1rem;
    font-size: .4rem;
  }
</style>
