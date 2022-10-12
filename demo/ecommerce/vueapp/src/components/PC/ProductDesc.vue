<template>
  <div class="product-desc-container">
    <!-- 图片放大 -->
    <div class="enlarge" v-show="isShow">
      <img class="image" @error="handleError" :src="data.image" alt="" />
      <i @click="handleOriginImg" class="iconfont icon-shanchu"></i>
    </div>
     <!-- 正统页面 -->
    <div class="desc-wrapper" :style="{ opacity: isShow ? 0 : 1 }">
      <!-- 左边图片部分 -->
      <div class="imgs left" ref="left" :class="{ 'left-error': isImgError }">
        <!-- <span v-if="isImgError" class="img-error">图片加载错误</span> -->
        <img class="image" @error="handleError" :src="data.image" alt="" />
        <i @click="handleEnlargeImg" class="iconfont icon-tupianfangda"></i>
      </div>
      <!-- 右边文字等详情 -->
      <div
        class="right"
        ref="right"
        :style="{ transform: `translateY(${getScrollTop}px)` }"
      >
        <!-- METASOUL DEV 字样 -->
        <div class="label letter-space gray-color small-fontsize">
          METASOUL DEV
        </div>
        <!-- title详情 -->
        <div class="title letter-space" :title="data.description">
          {{ data.description }}
        </div>
        <!-- 价格 -->
        <div class="price letter-space">
          <!-- 划掉的价格 -->
          <s class="small-fontsize gray-color" v-if="data.price&&data.price.split('-')[1]">{{
            data.price.split('-')[1]
          }}</s>
          <!-- 打折后的价格 -->
          <span class="discount" :class="{ nomargin: !data.price||!data.price.split('-')[1] }">{{
            data.price&&data.price.split('-')[0] ? data.price.split('-')[0] : '$39'
          }}</span>
          <!-- <s class="small-fontsize gray-color"
            >{{ data.price.split('-')[1] }}{{ data.price ? ' USD' : '' }}</s
          >
          <span class="discount"
            >{{ data.price.split('-')[0] }}{{ data.price ? ' USD' : '' }}</span
          > -->
          <a :href="data.url" class="sale">Sale</a>
        </div>
        <!-- 购买数量的加减 -->
        <div class="quantity-container">
          <div class="quantity-label letter-space gray-color small-fontsize">
            Quantity
          </div>
          <div class="quantity-button">
            <span @click="handleDeNum" class="de">-</span>
            <span @click="handleNumEdit" class="num">{{ num }}</span>
            <input
              ref="editInput"
              @blur="handleEditNumComplete"
              autofocus
              v-if="numEditShow"
              class="num-edit"
              v-model="editNum"
              type="text"
            />
            <span @click="handleInNum" class="in">+</span>
          </div>
        </div>
        <!-- 两个button -->
        <div class="add-card-button letter-space comment-button">
          Add to Card
        </div>
        <div class="buy-button letter-space comment-button">Buy it now</div>
      </div>
    </div>
  </div>
</template>

<script>
import defaultImg from '@/assets/default-img.webp';
export default {
  props: ['data'],
  data() {
    return {
      isShow: false,
      num: 1,
      numEditShow: false,
      editNum: '',
      isImgError: false,

      getScrollTop: 0,
      leftHeight: 0,
      rightHeight: 0,
    };
  },
  mounted() {
    this.getHeightInfo();
    window.addEventListener('resize', this.getHeightInfo);

    document.addEventListener('scroll', this.handleScroll);
  },
  methods: {
    handleError(e) {
      // this.isImgError = true;
      e.target.src = defaultImg;
    },
    handleScroll() {
      if (!this.$refs.left || !this.$refs.right) {
        return;
      }
      if (document.documentElement.clientWidth <= 912) {
        this.getScrollTop = 0;
        return;
      }
      if (this.$refs.left.clientWidth === this.$refs.right.clientWidth) {
        // 高相同，说明不在同一行
        this.getScrollTop = 0;
      } else if (
        this.$refs.left.clientHeight - this.$refs.right.clientHeight >=
        document.documentElement.scrollTop
      ) {
        this.getScrollTop = document.documentElement.scrollTop;
      }
    },
    getHeightInfo() {
      this.getScrollTop = 0;
      this.leftHeight = this.$refs.left.clientWidth * 1;
      this.rightHeight = this.leftHeight;
      this.$refs.left.style.height = this.$refs.left.clientWidth * 1 + 'px';

      // this.$refs.right.style.height = this.leftHeight +"px";
      // this.leftHeight = this.$refs.left.clientHeight;
      // this.rightHeight = this.$refs.right.clientHeight;
    },
    handleEnlargeImg() {
      this.isShow = true;
      this.$emit('enlarge', false);
    },
    handleOriginImg() {
      this.isShow = false;
      this.$emit('enlarge', true);
    },
    handleDeNum() {
      this.num--;
      if (this.num < 1) {
        this.num = 1;
      }
    },
    handleInNum() {
      this.num++;
    },
    handleNumEdit() {
      this.editNum = this.num;
      this.num = '';
      this.numEditShow = true;
      this.$nextTick(() => {
        this.$refs.editInput.focus();
      });
    },
    handleEditNumComplete() {
      this.num = this.editNum;
      this.editNum = '';
      this.numEditShow = false;
    },
  },
  beforeDestroy() {
    document.removeEventListener('scroll', this.handleScroll);
    window.removeEventListener('resize', this.getHeightInfo);
  },
  watch: {
    $route() {
      this.isImgError = false;
    },
  },
};
</script>

<style scoped lang="less">
@import url('//at.alicdn.com/t/c/font_3668378_4x8d2yrau7p.css');
.product-desc-container,
.desc-wrapper {
  width: 100%;
  display: flex;
  margin-bottom: 20px;
  box-sizing: border-box;
  position: relative;
}

.desc-wrapper {
  width: 100%;
}
.enlarge {
  width: 100%;
  position: absolute;
  left: 0;
  top: 0;
  z-index: 1000;
  font-size: 0;
}
 /* .enlarge {
   width: 100%;
   position: relative;
   font-size: 0;
 } */
.left {
  width: 100%;
   /* height: fit-content; */
  border: 1px solid lighten(#ccc, 10%);
  position: relative;
  font-size: 0;
  text-align: center;
  box-sizing: border-box;
}
.left-error {
  height: 100%;
}
.img-error {
  font-size: 20px;
  color: #ccc;
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 100%;
  text-align: center;
}
.image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.icon-tupianfangda {
  border: 1px solid #ccc;
  padding: 5px;
  border-radius: 50%;
  color: rgba(51, 51, 51, 0.8);
  position: absolute;
  left: 10px;
  top: 10px;
  cursor: pointer;
}
.icon-shanchu {
  border: 1px solid #ccc;
  padding: 5px;
  border-radius: 50%;
  color: rgba(51, 51, 51, 0.8);
  position: absolute;
  right: -10px;
  top: 0px;
  cursor: pointer;
}

.title {
  line-height: 2.5rem;
  font-size: 1.5rem;
  color: lighten(#000, 20%);

  display: -webkit-box;
  -webkit-box-orient: vertical;
  overflow: hidden;
  -webkit-line-clamp: 4;
}
.right {
  width: 70%;
  height: fit-content;
  padding-left: 2vw;
  box-sizing: border-box;
}

@media screen and (max-width: 912px) {
  .desc-wrapper {
    flex-direction: column;
  }
  .right {
    width: 100%;
    margin-top: 20px;
    padding-left: 0;
  }
}
.price {
  margin: 20px 0;
}
.discount {
  margin: 0px 5px;
}
.discount.nomargin {
  margin-left: 0;
}
.quantity-button {
  margin: 10px 0;
  border: 1px solid #333;
  width: 50%;

  /* display: flex;
  justify-content: space-around; */
  padding: 10px 0;
  margin-bottom: 20px;
  position: relative;
}
.quantity-button span {
  display: inline-block;
  width: 33.3%;
  text-align: center;
}
.in,
.de {
  cursor: pointer;
}
.num-edit {
  width: 20px;
  position: absolute;
  text-align: center;
  outline: none;
  padding: 20px;
  /* // top: -10px; */
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  opacity: 0.9;
  font-size: 16px;
}

/*  */
.letter-space {
  letter-spacing: 1px;
}
.gray-color {
  color: #666;
}
.small-fontsize {
  font-size: 10px;
}
.sale {
  padding: 5px 10px;
  border-radius: 15px;
  line-height: 1;
  background: rgb(51, 79, 180);
  color: rgb(255, 255, 255);
  font-size: 12px;
  cursor: pointer;
}
.comment-button {
  max-width: 100%;
  padding: 10px 0;
  border: 1px solid #000;
  margin: 10px 0;
  text-align: center;
  cursor: pointer;
}
.buy-button {
  background: #000;
  color: #fff;
}
a {
  text-decoration: none;
}
</style>
