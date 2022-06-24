<script lang="ts">
  import Classifier from "./lib/Classifier.svelte";
</script>

<main>
  <h1 id="main_header">Etymology Classifier</h1>
  <Classifier />
</main>

<style lang="scss">
  :global {
    body {
      background: #fefefe;
    }
  }

  :root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
      Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
  }

  main {
    padding: 1em;
    margin: 0;
    width: 90vw;
    #main_header {
      text-align: center;
    }
  }
</style>
