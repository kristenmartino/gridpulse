# Third-Party Notices

GridPulse depends on third-party data and assets. This file lists the
**runtime data sources** it fetches (and redistributes through the UI and
the public `/api/v1` endpoints) and the **bundled assets** it ships, each
with its license, where it's used, and where to find the original source.

## Runtime data sources

GridPulse fetches these feeds in the scoring/training jobs and
redistributes derived values through the dashboard and the public API.
Attribution travels with the API payloads (`attribution` field) so
downstream consumers inherit the terms below.

| Source | Used for | License / terms |
| --- | --- | --- |
| [U.S. Energy Information Administration — Form EIA-930](https://www.eia.gov/opendata/) | Hourly demand, generation, and interchange (the forecast target) | U.S. Government work — public domain. EIA requests citation of the agency as the source. |
| [Open-Meteo](https://open-meteo.com/) | 17 weather variables (the forecast features) — historical + forecast | **CC-BY-4.0.** Attribution required: "Weather data by Open-Meteo.com". |
| [NOAA / National Weather Service](https://www.weather.gov/) | Severe-weather alerts on the Risk tab | U.S. Government work — public domain. |

The commercial-use posture of the Open-Meteo free tier and the fuller
in-product attribution surface are tracked in
[#256](https://github.com/kristenmartino/gridpulse/issues/256).

## Bundled assets

### `assets/ba_polygons.geojson`

**Source**: filtered from
[`electricitymaps/electricitymaps-contrib`](https://github.com/electricitymaps/electricitymaps-contrib)
(`geo/world.geojson`, retrieved 2026-05-02). MIT-licensed.

**Filtering performed**: kept only US-prefixed features whose
EIA-930 respondent suffix matches one of the 51 BA codes in
`config.REGION_COORDINATES`. Properties replaced with
`{region: <our internal code>, name: <REGION_NAMES value>}`.

**License** (electricitymaps-contrib):

```
MIT License

Copyright (c) 2017 Electricity Maps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
