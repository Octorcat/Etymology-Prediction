<script lang="ts">
  import { spring } from "svelte/motion";
  import axios from "axios";
  import memoize from "memoizee";
  import PieChart from "./output/PieChart.svelte";
  import DataTable from "./output/DataTable.svelte";
  import type { Spring } from "svelte/motion";

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

  const getEtymology = async (word: string): Promise<Etymology> => {
    try {
      (loading = true), (error = "");
      if (word) {
        const res = await axios.get<ApiResponse>(`${API_SERVER_URL}/${word}`);
        return { ...res.data.etymology };
      } else {
        return DEFAULT_ETYMOLOGY;
      }
    } catch (err) {
      error =
        err.code == "ERR_BAD_REQUEST" ? err.response.data.message : err.message;
      return DEFAULT_ETYMOLOGY;
    } finally {
      loading = false;
    }
  };

  const memoGetEtymology: (word: string) => Promise<Etymology> =
    memoize(getEtymology);

  const handleChange = async (evt: Event): Promise<void> => {
    try {
      const word = (evt.target as HTMLInputElement).value;
      etymology = await memoGetEtymology(word);
    } catch (error) {
      console.error(error);
    }
  };

  let timeoutID: number = -1;
  const debounce = (callback: (evt: Event) => void, time: number): void => {
    clearTimeout(timeoutID);
    timeoutID = setTimeout(callback, time);
  };

  const debouncedHandleChange = (evt: Event): void =>
    debounce(() => handleChange(evt), 600);

  const format = (probability: number): string =>
    probability < 0.1
      ? "<0%"
      : probability > 0.99
      ? ">99%"
      : `~${Math.round(100 * probability)}%`;

  $: pieCutPercent = 100 * (word ? etymology.Latin : DEFAULT_ETYMOLOGY.Latin);
  $: store.set(pieCutPercent);
</script>

<section id="classifier">
  <input
    bind:value={word}
    on:input={debouncedHandleChange}
    placeholder="Enter word here"
    id="word"
  />
  <section id="results">
    {#if loading}
      <p id="loading-message">...waiting</p>
    {:else if error}
      <p id="error-message">{error}</p>
    {:else}
      <section id="response">
        <PieChart size={200} percent={$store} />
        <div id="percentage-probs">
          <span>{format(etymology.Germanic)} Germanic</span>
          <span>{format(etymology.Latin)} Latin</span>
        </div>
        <DataTable {...etymology} />
      </section>
    {/if}
  </section>
</section>

<style lang="scss">
  @mixin flex-stack {
    display: flex;
    flex-direction: column;
    justify-self: center;
    align-items: center;
  }

  @mixin flex-center {
    display: flex;
    justify-self: center;
    align-items: center;
  }

  section#classifier {
    @include flex-stack;
    gap: 20px;
  }

  input#word {
    font-size: x-large;
    width: min(80vw, 310px);
  }

  section#results {
    font-size: x-large;
    width: min(80vw, 310px);
    p#loading-message {
      text-align: center;
    }
    p#error-message {
      color: red;
    }
    section#response {
      @include flex-stack;
      gap: 20px;
      div#percentage-probs {
        @include flex-center;
      }
    }
  }
</style>
