<script lang="ts">
  export let Germanic: number = 0.5;
  export let Latin: number = 0.5;
</script>

<section id="raw-results_table">
  <table>
    <tr>
      <th>Etymology</th>
      <th>Prob</th>
    </tr>
    <tr>
      <td>Germanic</td>
      <td>{Germanic}</td>
    </tr>
    <tr>
      <td>Latin</td>
      <td>{Latin}</td>
    </tr>
  </table>
</section>

<style>
  #raw-results_table {
    width: min(80vw, 300px);
    overflow-x: scroll;
    border: 1px solid purple;
  }
</style>
