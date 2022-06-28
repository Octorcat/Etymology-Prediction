<script lang="ts">
  const currentYear: number = new Date().getFullYear();
</script>

<footer id="main_footer">
  <a id="author_link" href="https://www.frederickroman.com">
    &copy; Frederick Roman {currentYear}
  </a>
</footer>

<style lang="scss">
  #main_footer {
    text-align: center;
    margin: 20px auto;
    #author_link {
      text-decoration: none;
    }
  }
</style>
