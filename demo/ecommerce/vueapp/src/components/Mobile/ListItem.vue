<template>
  <router-link
    v-if="data.item_id"
    :to="{ name: 'product', params: { productId: data.item_id } }"
    class="list-item-container"
  >
    <div class="inner">
      <span v-if="imgError" class="img-error">图片加载错误</span>
      <img
        v-lazy="data.image"
        @load="handleLoad"
        @error="handleImgError"
        class="img"
        alt=""
      />
      <span class="sale">Sale</span>
    </div>
    <div class="information">
      <div class="desc" :style="{ color: acolor }">
        {{ data.description }}
        {{
          data.description
            ? ''
            : 'Doublju companyservices to customer qualified products withqualified products with'
        }}
      </div>
      <div class="price">
        <s v-if="underline && data.price && data.price.split('-')[1]">{{
          data.price.split('-')[1]
        }}</s>
        <span :style="{ color: acolor }" v-if="data.price">{{ data.price.split('-')[0] }}</span>
        <span :style="{ color: acolor }" v-if="!data.price">$39</span>
        <!-- <s v-if="underline && data.price"
          >{{
            data.price.split('-')[1]
              ? data.price.split('-')[1].split('.')[0]
              : data.price.split('-')[1]
          }}{{ data.price ? ' USD' : '' }}</s
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
      imgError: false
    };
  },
  methods: {
    handleLoad() {
      this.isShow = true;
    },
    handleImgError(e) {
      // let span = document.createElement('span');
      // span.classList.add("img-error");
      // span.innerText = '图片加载错误';
      // e.detail.el.parentNode.appendChild(span);
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
  width: 50%;
  display: flex;
  flex-direction: column;
  justify-content: end;
  cursor: pointer;
  padding: 0.2rem;
  box-sizing: border-box;
  border: 1px solid #ccc;
  border-color: lighten(#ccc, 10%);
  border-top: 0;
}
@media screen and (max-width: 700px) {
  .list-item-container {
    width: 50%;
  }
}
.inner {
  width: 100%;
  height: 5rem;
  overflow: hidden;
  position: relative;

  // font-size: .5rem;
  // color: #ccc;
  // text-align: center;
}
.img-error {
  font-size: 0.5rem;
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
  padding: 0.2rem 0.25rem;
  border-radius: 0.5rem;
  line-height: 1;
  background: rgb(51, 79, 180);
  position: absolute;
  left: 0.1rem;
  bottom: 0.1rem;
  color: rgb(255, 255, 255);
  font-size: 0.25rem;
  letter-spacing: 0.02rem;
}
.img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center center;
  // object-position: 0px 0px;
  cursor: pointer;
  transition: all 0.5s;
}
.img:hover {
  transform: scale(1.03);
}
.information {
  font-size: 0.7rem;
}
.desc,
.price {
  margin: 10px 0;
  letter-spacing: 0.05rem;
  font-size: 0.3rem;
}
.desc {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  overflow: hidden;
  -webkit-line-clamp: 3;

  color: lighten(#000, 10%);
}
.price {
  letter-spacing: 0.05rem;
}
.desc:hover {
  text-decoration: underline;
}
.price span {
  // font-size: .4rem;
  color: rgba(0, 0, 0, 0.9);
}
s {
  color: rgba(102, 102, 102, 0.8);
  margin-right: 10px;
}
</style>
