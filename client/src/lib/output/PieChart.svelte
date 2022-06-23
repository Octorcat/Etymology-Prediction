<script lang="ts">
  /**
   * This componenet was taken (with some modifications) from 
   * the Pie Chart (svelte/motion) example on the official Svelete page
   * @see {@link https://svelte.dev/repl/e68bcf378d2c410d9c123fd309fc6dbb?version=3.42.4}
   * And it is used under MIT license.
   * @see {@link https://github.com/sveltejs/svelte/blob/master/LICENSE.md}
   * by Svelte contributors (these people)
   * @see {@link https://github.com/sveltejs/svelte/graphs/contributors}
   *
   * **********************************************************************
   * 
   * Copyright (c) 2016-22 these people 
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy of
   * this software and associated documentation files (the "Software"), to deal in 
   * the Software without restriction, including without limitation the rights to use, 
   * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
   * Software, and to permit persons to whom the Software is furnished to do so, 
   * subject to the following conditions:
   * 
   * The above copyright notice and this permission notice shall be included in 
   * all copies or substantial portions of the Software.
   * 
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
   * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
   * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
   * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
   * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
   * DEALINGS IN THE SOFTWARE.
   * 
   * **************************************************************************
   * 
   * Frederick Romnan:
   * - Changed the colors
   * - Change the orientation
   * 
   */
  export let size: number = 200;
  export let percent: number = 0;
  export let bgColor: string = "#11e";
  export let fgColor: string = "#e11";

  $: viewBox = `0 0 ${size} ${size}`;

  $: radius = size / 2;
  $: halfCircumference = Math.PI * radius;
  $: pieSize = halfCircumference * (percent / 100);
  $: dashArray = `${0.5*pieSize} ${halfCircumference - pieSize} ${pieSize}`;
</script>

<svg width={size} height={size} {viewBox}>
  <circle r={radius} cx={radius} cy={radius} fill={bgColor} />
  <circle
    r={radius / 2}
    cx={radius}
    cy={radius}
    fill={bgColor}
    stroke={fgColor}
    stroke-width={radius}
    stroke-dasharray={dashArray}
  />
</svg>
