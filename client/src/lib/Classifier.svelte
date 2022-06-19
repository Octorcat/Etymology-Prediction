<script lang="ts">
  interface Etymology {
    Germanic: number;
    Latin: number;
  }

  const DEFAULT_ETYMOLOGY: Readonly<Etymology> = Object.freeze({
    Germanic: 0.5,
    Latin: 0.5,
  });
  const API_SERVER_URL = "http://localhost:5000/etymology/";
  let promiseEtymology: Promise<Etymology> = getEtymology("");

  const handleChange = (evt: Event): void => {
    const word = (evt.target as HTMLInputElement).value;
    promiseEtymology = getEtymology(word);
  };

  async function getEtymology(word: string): Promise<Etymology> {
    if (!word) return DEFAULT_ETYMOLOGY;
    const res = await fetch(`${API_SERVER_URL}${word}`);
    const json = await res.json();
    if (res.ok) {
      return json.etymology;
    } else {
      throw new Error(json);
    }
  }
</script>

<section>
  <input on:input={handleChange} placeholder="Enter word here" />
  {#await promiseEtymology}
    <p>...waiting</p>
  {:then { Germanic, Latin }}
    <p>Germanic {Germanic}</p>
    <p>Latin {Latin}</p>
  {:catch error}
    <p style="color: red">{error.message}</p>
  {/await}
</section>

<style>
</style>
