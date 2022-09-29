<template>
  <div id="app">
    <!-- <HomeView v-if="$route.name==='home'"></HomeView> -->
    <!-- <router-view v-else-if="$route.name==='product'"></router-view> -->
    <router-view></router-view>
  </div>
</template>

<script>
// import HomeView from '@/views/HomeView.vue';
export default {
  components: {
    // HomeView,
  },
  created() {
    // console.log(this.$route);
  }
};
</script>

<style>
</style>
