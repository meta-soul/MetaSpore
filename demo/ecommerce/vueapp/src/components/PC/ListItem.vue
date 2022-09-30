<template>
  <router-link
    v-if="data.item_id"
    :to="{
      name: 'product',
      params: { productId: data.item_id, userId: $store.state.cur_user },
    }"
    class="list-item-container"
  >
    <div class="inner">
      <span v-if="imgError" class="img-error">图片加载错误</span>
      <img
        v-lazy="data.image"
        @load="handleLoad"
        @error="handleError"
        class="img"
        alt=""
      />
      <span class="sale">Sale</span>
    </div>
    <div class="information">
      <div class="desc" :style="{ color: acolor }">
        {{ data.description
        }}{{
          data.description
            ? ''
            : 'Doublju company services to customer qualified products withqualified products with'
        }}
      </div>
      <div class="price">
        <s v-if="underline">{{ data.price.split('-')[1] }}</s>
        <span :style="{ color: acolor }">{{ data.price.split('-')[0] }}</span>
        <span :style="{ color: acolor }" v-if="!data.price">$39</span>
        <!-- <s v-if="underline"
          >{{ data.price.split('-')[1] }}{{ data.price ? ' USD' : '' }}</s
        >
        <span :style="{ color: acolor }"
          >{{ data.price.split('-')[0] }}{{ data.price ? ' USD' : '' }}</span
        >
        <span :style="{ color: acolor }" v-if="!data.price">$39 USD</span> -->
      </div>
    </div>
  </router-link>
</template>

<script>
export default {
  props: ['data', 'underline', 'acolor'],
  data() {
    return {
      isShow: false,
      imgError: false,
    };
  },
  methods: {
    handleLoad() {
      this.isShow = true;
    },
    handleError(e) {
      // e.detail.el.parentNode.parentNode.remove();
      this.imgError = true;
      e.detail.el.remove();
    },
  },
};
</script>

<style scoped lang="less">
a {
  text-decoration: none;
}
.list-item-container {
  width: 25%;
  display: flex;
  flex-direction: column;
  justify-content: end;
  cursor: pointer;
  border: 1px solid lighten(#ccc, 10%);
  padding: 1vw;
  box-sizing: border-box;
  border-top: 0;
}
@media screen and (max-width: 912px) {
  .list-item-container {
    width: 50%;
  }
}
.inner {
  width: 100%;
  height: 20vw;
  overflow: hidden;
  position: relative;
}
.img-error {
  // font-size: 0.5rem;
  font-size: 20px;
  color: #ccc;
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 100%;
  text-align: center;
}
.desc {
  width: 100%;
}
.sale {
  padding: 5px 10px;
  border-radius: 15px;
  line-height: 1;
  background: rgb(51, 79, 180);
  position: absolute;
  left: 10px;
  bottom: 10px;
  color: rgb(255, 255, 255);
  font-size: 12px;
}
.img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center center;
  /* object-position: 0px 0px; */
  cursor: pointer;
  transition: all 0.5s;
}
.img:hover {
  transform: scale(1.03);
}
.information {
  font-size: 0.7vw;
}
.desc,
.price {
  margin: 10px 0;
  font-family: Assistant, sans-serif;
  font-style: normal;
  letter-spacing: 2px;
}
.desc {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  overflow: hidden;
  -webkit-line-clamp: 3;
  color: #666;
}
.desc:hover {
  text-decoration: underline;
}
.price span {
  font-size: 1vw;
  color: rgba(0, 0, 0, 0.7);
}
s {
  color: #666;
  margin-right: 10px;
}
</style>
