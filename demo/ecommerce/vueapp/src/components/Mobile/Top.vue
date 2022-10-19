<template>
    <div v-if="isShow" @touchend.stop="handleClick" class="to-top"><i class="el-icon-top"></i></div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        isShow: false,
      };
    },
    created() {
      this.$bus.$on("mainScroll", this.handleScrollChange);
    },
    methods: {
      handleClick() {
        document.documentElement.scrollTo({
          top: 0,
          behavior: 'smooth',
        });
      },
      handleScrollChange(dom) {
        if (dom.scrollY > 300) {
          this.isShow = true;
        } else {
          this.isShow = false;
        }
      },
    },
    beforeDestroy() {
      this.$bus.$off("mainScroll", this.handleScrollChange);
    },
  };
  </script>
  
  <style lang="less" scoped>
  .to-top {
    text-align: center;
    cursor: pointer;
    position: fixed;
    bottom: 1.5rem;
    right: .05rem;
  }
  .el-icon-top {
    color: lighten(gray, 10%);
    font-size: .7rem;
    animation: identifier 1s linear infinite alternate;
}
@keyframes identifier {
    0% {
        transform: translateY(0);
    }
    100% {
        transform: translateY(20px);
    }
}
  </style>
  