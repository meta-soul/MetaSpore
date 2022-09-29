<template>
  <div class="product-desc-container" ref="container">
    <!-- 图片放大 -->
    <div class="enlarge" v-if="isShow">
      <img class="image" :src="data.image" alt="" />
      <i @click="handleOriginImg" class="iconfont icon-shanchu"></i>
    </div>
    <!-- 正统页面 -->
    <div class="desc-wrapper" v-else>
      <!-- 左边图片部分 -->
      <div class="imgs left" ref="left" v-if="!isImgError">
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
        <div
          ref="title"
          class="title elipse letter-space"
          :class="{ 'multiline-ellipsis': !isReadMore }"
          :title="data.description"
        >
          {{ data.description }}
        </div>
        <div
          ref="titleOut"
          class="title title-out letter-space"
          :title="data.description"
        >
          {{ data.description }}
        </div>
        <button v-if="readMoreShow" @click="handleReadMore" class="read-more">
          {{ isReadMoreValue }}
        </button>
        <!-- 价格 -->
        <div class="price letter-space">
          <!-- 划掉的价格 -->
          <s class="small-fontsize gray-color">{{
            data.price.split('-')[1]
          }}</s>
          <!-- 打折后的价格 -->
          <span class="discount">{{ data.price.split('-')[0] }}</span>
          <!-- 划掉的价格
          <s class="small-fontsize gray-color"
            >{{ data.price.split('-')[1] }}{{ data.price ? ' USD' : '' }}</s
          >
          打折后的价格
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
export default {
  props: ['data'],
  data() {
    return {
      isShow: false,
      num: 1,
      numEditShow: false,
      editNum: '',
      isReadMore: false,
      isReadMoreValue: '查看更多 >>>',
      readMoreTop: 0,
      readMoreShow: true,
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

    // console.log(this.$refs.title.clientHeight, this.$refs.titleOut.clientHeight);
    if(this.$refs.title.clientHeight === this.$refs.titleOut.clientHeight) {
      this.readMoreShow = false;
    }
  },
  methods: {
    handleError(e) {
      this.isImgError = true;
    },
    handleScroll() {
      if(!this.$refs.left || !this.$refs.right) {
        return;
      }
      if (
        this.$refs.left.clientWidth === this.$refs.right.clientWidth
      ) {
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
      this.leftHeight = this.$refs.left.clientHeight;
      this.rightHeight = this.$refs.right.clientHeight;
    },
    handleReadMore() {
      if (this.isReadMoreValue === '查看更多 >>>') {
        // 在点之前记录一下scrollTop的值
        this.readMoreTop = document.documentElement.scrollTop;
      } else if (this.isReadMoreValue === '收起 >') {
        document.documentElement.scrollTo({
          top: this.readMoreTop,
          behavior: 'smooth',
        });
      }
      this.isReadMore = !this.isReadMore;
      this.isReadMoreValue =
        this.isReadMoreValue === '收起 >' ? '查看更多 >>>' : '收起 >';
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
    document.addEventListener('scroll', this.handleScroll);
    window.addEventListener('resize', this.getHeightInfo);
  },
};
</script>

<style scoped lang="less">
@import url('//at.alicdn.com/t/c/font_3668378_4x8d2yrau7p.css');
.product-desc-container,
.desc-wrapper {
  width: 100%;
  /* display: flex; */
  margin-bottom: 1rem;
  box-sizing: border-box;
  position: relative;
}

.desc-wrapper {
  width: 100%;
}
.enlarge {
  width: 100%;
  position: relative;
  font-size: 0;
}
.left {
  width: 100%;
  height: fit-content;
  border: 1px solid lighten(#ccc, 10%);
  position: relative;
  font-size: 0;
}
.image {
  width: 100%;
  object-fit: cover;
}
.icon-tupianfangda {
  border: 0.02rem solid #ccc;
  padding: 0.2rem;
  border-radius: 50%;
  color: rgba(51, 51, 51, 0.8);
  position: absolute;
  left: 0.2rem;
  top: 0.2rem;
  cursor: pointer;
  font-size: 0.4rem;
}
.icon-shanchu {
  border: 0.02rem solid #ccc;
  padding: 0.2rem;
  border-radius: 50%;
  color: rgba(51, 51, 51, 0.8);
  position: absolute;
  right: 0.2rem;
  top: 0px;
  cursor: pointer;
  font-size: 0.4rem;
}

.title {
  line-height: 1rem;
  font-size: 0.7rem;
  margin-top: 0.2rem;
}
.title-out {
  opacity: 0;
  position: absolute;
  left: 0;
  top: 0;
  z-index: -100;
}

.read-more {
  border: 0;
  background: #fff;
  padding: 0;
  margin: 0;
  color: blue;

  font-size: 0.3rem;
}
.right {
  width: 50%;
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
  margin: 0.3rem 0;
  display: flex;
  align-items: center;
}
.discount {
  font-size: 0.4rem;
  margin: 0px 0.2rem;
}
.quantity-button {
  margin: 0.2rem 0;
  border: 1px solid #333;
  width: 4rem;
  font-size: 0.5rem;

  /* display: flex;
  justify-content: space-around; */
  padding: 0.2rem 0;
  margin-bottom: 0.5rem;
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
  width: 1rem;
  position: absolute;
  text-align: center;
  outline: none;
  padding: 0.5rem;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  opacity: 0.9;
  font-size: 0.5rem;
}

/*  */
.letter-space {
  letter-spacing: 0.05rem;
}
.gray-color {
  color: #666;
}
.small-fontsize {
  font-size: 0.3rem;
}
.sale {
  padding: 0.2rem 0.25rem;
  border-radius: 0.5rem;
  line-height: 1;
  background: rgb(51, 79, 180);
  color: rgb(255, 255, 255);
  font-size: 0.25rem;
  cursor: pointer;
}
.comment-button {
  max-width: 100%;
  padding: 0.2rem 0;
  border: 1px solid #000;
  margin: 0.25rem 0;
  text-align: center;
  cursor: pointer;
  font-size: 0.5rem;
}
.buy-button {
  background: #000;
  color: #fff;
}
a {
  text-decoration: none;
}
.multiline-ellipsis {
  /* 多行省略 */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  overflow: hidden;
  -webkit-line-clamp: 5;
}
</style>
