<template>
    <div v-if="isShow" @click="handleClick" class="to-top"><i class="el-icon-top"></i></div>
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
    width: 30px;
    height: 30px;
    border-radius: 50%;
    /* background: rgb(104, 172, 236); */
    color: #fff;
    text-align: center;
    cursor: pointer;
    position: fixed;
    bottom: calc(5vw);
    right: calc(8vw);
    line-height: 30px;
}
.el-icon-top {
    color: gray;
    font-size: 25px;
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
  