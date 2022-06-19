<script lang="ts">
  import Classifier from "./lib/Classifier.svelte";
</script>

<main>
  <h1> Etymology Classifier</h1>
  <Classifier />
</main>

<style>
  :root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
      Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
  }

  main {
    text-align: center;
    padding: 1em;
    margin: 0 auto;
  }
</style>
