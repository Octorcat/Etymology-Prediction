<script lang="ts">
  import { Spring, spring } from "svelte/motion";
  import PieChart from "./output/PieChart.svelte";
  import DataTable from "./output/DataTable.svelte";
  import axios from "axios";

  interface Etymology {
    Germanic: number;
    Latin: number;
  }

  interface ApiResponse {
    etymology: Etymology;
  }

  const DEFAULT_WORD: string = "";
  const DEFAULT_ETYMOLOGY: Readonly<Etymology> = Object.freeze({
    Germanic: 0.5,
    Latin: 0.5,
  });
  const API_SERVER_URL: string = "http://localhost:5000/etymology";
  const store: Spring<number> = spring(0, { stiffness: 0.3, damping: 0.3 });

  let loading: boolean = false;
  let error: string = "";
  let word: string = DEFAULT_WORD;
  let etymology: Etymology = DEFAULT_ETYMOLOGY;

  const getEtymology = async (word: string): Promise<void> => {
    try {
      (loading = true), (error = "");
      if (word) {
        const res = await axios.get<ApiResponse>(`${API_SERVER_URL}/${word}`);
        etymology = { ...res.data.etymology };
      } else {
        etymology = DEFAULT_ETYMOLOGY;
      }
    } catch (err) {
      error =
        err.code == "ERR_BAD_REQUEST" ? err.response.data.message : err.message;
      etymology = DEFAULT_ETYMOLOGY;
    } finally {
      loading = false;
    }
  };

  const handleChange = async (evt: Event): Promise<void> => {
    const word = (evt.target as HTMLInputElement).value;
    await getEtymology(word).catch(console.error);
  };

  const format = (probability: number): string =>
    probability < 0.1
      ? "<0%"
      : probability > 0.99
      ? ">99%"
      : `~${Math.round(100 * probability)}%`;

  $: pieCutPercent = 100 * (word ? etymology.Latin : DEFAULT_ETYMOLOGY.Latin);
  $: store.set(pieCutPercent);
</script>

<section id="classifier_section">
  <input
    bind:value={word}
    on:input={handleChange}
    placeholder="Enter word here"
    id="word_input"
  />
  <section id="classification-results">
    {#if loading}
      <p id="await-msg">...waiting</p>
    {:else if error}
      <p style="color: red">{error}</p>
    {:else}
      <section id="classification-response">
        <PieChart size={200} percent={$store} />
        <div id="classification-percentage-probs">
          <span>{format(etymology.Germanic)} Germanic</span>
          <span>{format(etymology.Latin)} Latin</span>
        </div>
        <DataTable {...etymology} />
      </section>
    {/if}
  </section>
</section>

<style>
  #classifier_section {
    display: flex;
    flex-direction: column;
    justify-self: center;
    align-items: center;
    gap: 20px;
  }

  #classification-response {
    display: flex;
    flex-direction: column;
    justify-self: center;
    align-items: center;
    gap: 20px;
  }

  #word_input {
    font-size: x-large;
    width: min(80vw, 310px);
  }

  #await-msg {
    text-align: center;
  }

  #classification-results {
    font-size: x-large;
    width: min(80vw, 310px);
  }

  #classification-percentage-probs {
    display: flex;
    justify-self: center;
    align-items: center;
  }
</style>
