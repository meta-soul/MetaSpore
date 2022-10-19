<template>
  <div class="list-group-container">
    <div class="title"><span class="title-title">{{ datas.title }}</span></div>
    <div class="content" v-loading="!datas.items">
      <ListItem
        :underline="underline"
        :acolor="acolor"
        :data="item"
        v-for="(item, i) in datas.items"
        :key="i"
      />
    </div>
    <!-- <Top /> -->
  </div>
</template>

<script>
import ListItem from './ListItem.vue';
// import Top from '@/components/PC/Top.vue';
export default {
  props: ['datas', 'underline', 'acolor'],
  components: {
    ListItem,
    // Top
  },
  
  // mounted() {
  //   window.addEventListener("scroll", this.handleScroll);
  // },
  // methods: {
  //   handleScroll() {
  //     this.$bus.$emit("mainScroll", window);
  //   },
  //   beforeDestroy() {
  //     window.removeEventListener("scroll", this.handleScroll);
  //   }
  // }
};
</script>

<style scoped lang="less">
.list-group-container {
  padding-bottom: 10px;
}
.title {
  font-family: Assistant, sans-serif;
  font-style: normal;
  letter-spacing: 2px;
  color: lighten(#000, 40%);
  font-size: 25px;
  width: fit-content;


  text-shadow: 
              -1px -1px #fff;
  /* animation: identifier 1s linear infinite alternate; */
}
@keyframes identifier {
  0% {
    transform: scale(1);
  }
  100% {
    transform: scale(1.1);
  }
}
.content {
  width: 100%;
  /* display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr; */

  display: flex;
  flex-wrap: wrap;
  margin-top: 20px;
  /* grid-row-gap: 10px; */
   padding: 20px 0;
  box-sizing: border-box;
}
@media screen and (min-width: 912px) {
  .content a:nth-of-type(4n),
  .content a:nth-of-type(4n-2),
  .content a:nth-of-type(4n-1) {
    border-left: 0;
    /* border: 1px solid #008c8c; */
  }
  .content a:nth-of-type(1),
  .content a:nth-of-type(2),
  .content a:nth-of-type(3),
  .content a:nth-of-type(4) {
    border-top: 1px solid lighten(#ccc, 10%);
  }
  /* // .content a:last-of-type {
  //   border-right: 1px solid lighten(#ccc, 10%);
  // } */
}

@media screen and (max-width: 912px) {
  .content a:nth-of-type(2n) {
    border-left: 0;
  }
  .content a:nth-of-type(1) {
    border-top: 1px solid lighten(#ccc, 10%);
  }
  .content a:nth-of-type(2) {
    border-top: 1px solid lighten(#ccc, 10%);
  }
}

/* @media screen and (max-width: 900px) {
  .content {
    grid-template-columns: 1fr 1fr;
  }
} */
</style>
