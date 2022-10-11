<template>
  <div class="list-group-container">
    <div class="title">{{ datas.title }}</div>
    <div class="content" v-loading="!datas.items">
      <ListItem
        :underline="underline"
        :acolor="acolor"
        :data="item"
        v-for="(item, i) in datas.items"
        :key="i"
      />
    </div>
  </div>
</template>

<script>
import ListItem from './ListItem.vue';
export default {
  props: ['datas', 'underline', 'acolor'],
  components: {
    ListItem,
  },
};
</script>

<style scoped lang="less">
.list-group-container {
  padding-bottom: 1rem;
}
.title {
  letter-spacing: 0.05rem;
  font-size: 0.5rem;
  color: lighten(#000, 40%);
}
.content {
  width: 100%;
  /* display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr; */

  display: flex;
  flex-wrap: wrap;
  margin-top: 0.5rem;
  /* grid-row-gap: 10px; */
  box-sizing: border-box;
}

.content a:nth-of-type(2n) {
  border-left: 0;
}
.content a:nth-of-type(1) {
  border-top: 1px solid lighten(#ccc, 10%);
}
.content a:nth-of-type(2) {
  border-top: 1px solid lighten(#ccc, 10%);
}
/* .content a:nth-of-type(4n) {
  border-top: 0;
  border: 1px solid #008c8c;
}
.content a:nth-of-type(4n-1) {
  border-top: 0;
  border: 1px solid #008c8c;
} */

/* @media screen and (max-width: 900px) {
  .content {
    grid-template-columns: 1fr 1fr;
  }
} */
</style>
