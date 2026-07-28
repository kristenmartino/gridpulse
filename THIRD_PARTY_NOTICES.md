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

> **Licence correction, 2026-07-28.** This asset was previously described here
> as MIT-licensed. That was wrong. The upstream repository relicensed to
> **AGPL-3.0** in January 2023, and the geometries we use were edited after
> that date, so the MIT grant does not cover them. Corrected below; the
> replacement work is tracked in
> [#357](https://github.com/kristenmartino/gridpulse/issues/357).

**Source**: filtered from
[`electricitymaps/electricitymaps-contrib`](https://github.com/electricitymaps/electricitymaps-contrib)
(`geo/world.geojson`, retrieved 2026-05-02).

**License**: **AGPL-3.0** (GNU Affero General Public License v3.0).

The upstream repository carries two licence files, and which one applies is a
question of *when* a contribution was made, not which file you read first:

- `LICENSE.md` — AGPL-3.0. The upstream README states: *"This repository is
  licensed under GNU-AGPLv3 since v1.5.0 … Contributions prior to commit
  `cb9664f` were licensed under MIT license"*.
- `LICENSE_MIT.txt` — MIT, `Copyright (c) Jan 2020 - Jan 2023 Tomorrow`.
  Commit `cb9664f` landed **2023-01-30**.

**Why the MIT grant does not cover this asset.** The file's history lives at
`web/geo/world.geojson` (the current `geo/` path is a 2026-01-13 copy, so a
path-scoped history query on it is misleading). That original path has 46
commits, **23 of them after the 2023-01-30 relicence**, including commit
`83cfc4fe` (2023-07-13), *"Changes borders of El Paso and ERCOT"* — EPE and
ERCOT are both in our 51-BA set — and `67ca5518` (2024-07-16), *"clean up
world and round to 3 decimals"*, which rewrites coordinates file-wide. The
version retrieved on 2026-05-02 therefore contains post-relicence
contributions to the geometries we ship.

**Obligations this carries.** AGPL-3.0 §13 extends copyleft to works made
available over a network. GridPulse serves a choropleth derived from this
asset from a public web tier. Nothing in this repository is currently
structured to meet those obligations, which is the substance of
[#357](https://github.com/kristenmartino/gridpulse/issues/357) — this notice
records the licence accurately; it does not by itself establish compliance.

**Not legal advice.** Two questions remain genuinely open and are not resolved
here: whether coordinate data of this kind attracts copyright at all (as
against the selection and arrangement of it), and what compliance would
require in practice. Both need qualified advice before any commercial use.

**Filtering performed**: kept only US-prefixed features whose
EIA-930 respondent suffix matches one of the 51 BA codes in
`config.REGION_COORDINATES`. Properties replaced with
`{region: <our internal code>, name: <REGION_NAMES value>}`.

**Upstream licence texts**, reproduced for reference. The MIT text is retained
because it governs pre-2023-01-30 contributions, not because it governs this
asset:

*AGPL-3.0* — full text at
<https://github.com/electricitymaps/electricitymaps-contrib/blob/master/LICENSE.md>
(GNU Affero General Public License v3.0, 19 November 2007).

*MIT (pre-`cb9664f` contributions only)*:

```
MIT License

Copyright (c) Jan 2020 - Jan 2023 Tomorrow

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
