# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                  |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------- | -------: | -------: | ------: | --------: |
| components/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| components/\_callbacks\_alerts.py     |      183 |       17 |     91% |392-394, 402-403, 480-485, 502-504, 535, 609-610, 653-654 |
| components/\_callbacks\_backtest.py   |      295 |       44 |     85% |95, 103, 105, 139-156, 174-177, 212, 223, 241-242, 544-549, 557-558, 617-620, 662-664, 674-676, 757-758 |
| components/\_callbacks\_forecast.py   |      656 |      123 |     81% |366, 429-430, 532, 545-548, 563-566, 594-597, 610-615, 634-639, 659-664, 700-701, 717, 745-746, 757, 766-778, 863, 1200, 1215-1217, 1347-1349, 1366-1368, 1382-1394, 1430-1432, 1443-1448, 1471, 1475-1476, 1482, 1488-1490, 1562, 1590-1594, 1690-1691, 1727-1728, 1832-1850, 1874-1927 |
| components/\_callbacks\_generation.py |       72 |        4 |     94% |74, 99-100, 112 |
| components/\_callbacks\_models.py     |      349 |       19 |     95% |92, 182, 641, 692, 848, 956-961, 975-977, 994-997, 1037, 1109-1110 |
| components/\_callbacks\_overview.py   |      899 |      213 |     76% |155, 180-182, 277-279, 480, 487, 495, 608, 658, 662, 666, 668, 677, 690-700, 756, 788, 851, 917, 926, 996-1086, 1090-1093, 1097, 1145, 1155, 1192, 1317, 1339, 1364-1373, 1387, 1403-1454, 1708, 1749-1808, 1933-1978, 2012, 2048, 2085, 2087, 2095, 2162, 2240-2243, 2305-2306, 2313-2318, 2326-2337, 2340-2342, 2347-2379, 2403-2405, 2407-2411, 2503-2507, 2701-2740 |
| components/\_callbacks\_shared.py     |      235 |       19 |     92% |380-388, 405-406, 411-412, 618-619, 626-627, 742, 744 |
| components/\_callbacks\_us\_grid.py   |      302 |       46 |     85% |448, 634, 797, 926-982, 994-1002, 1056, 1059 |
| components/\_callbacks\_weather.py    |       42 |        0 |    100% |           |
| components/accessibility.py           |       23 |        3 |     87% |124, 141-142 |
| components/callbacks.py               |      360 |       84 |     77% |152-154, 232, 372-377, 525-573, 749, 789-790, 824-852, 868-882, 899-912, 921-933, 960-961 |
| components/cards.py                   |      107 |        6 |     94% |275-278, 308, 451 |
| components/error\_handling.py         |      106 |       13 |     88% |52, 99, 248-253, 315, 366-370, 454, 469 |
| components/icons.py                   |       14 |        1 |     93% |       123 |
| components/insights.py                |      351 |        1 |     99% |       467 |
| components/layout.py                  |       29 |        1 |     97% |        78 |
| components/tab\_alerts.py             |       15 |        0 |    100% |           |
| components/tab\_demand\_outlook.py    |       41 |        0 |    100% |           |
| components/tab\_models.py             |       14 |        0 |    100% |           |
| components/tab\_overview.py           |        4 |        0 |    100% |           |
| components/tab\_us\_grid.py           |        9 |        0 |    100% |           |
| data/\_\_init\_\_.py                  |        7 |        0 |    100% |           |
| data/ai\_briefing.py                  |      193 |       82 |     58% |77-81, 100-135, 204, 213, 216-217, 228-232, 237-242, 245, 258-269, 274-275, 280, 293-295, 300, 303, 350-364, 369-379, 386-410, 423 |
| data/audit.py                         |       54 |        0 |    100% |           |
| data/cache.py                         |      102 |        2 |     98% |  193, 205 |
| data/demo\_data.py                    |       89 |        0 |    100% |           |
| data/eia\_client.py                   |      240 |        2 |     99% |  327, 329 |
| data/explainability.py                |       44 |        3 |     93% |   204-206 |
| data/feature\_engineering.py          |      192 |       11 |     94% |59-60, 184, 225, 373, 551-552, 572-573, 731, 736 |
| data/forecast\_history.py             |      137 |       11 |     92% |54-55, 116-117, 149, 157, 181-183, 420-421 |
| data/gcs\_store.py                    |       71 |        3 |     96% |31, 52, 99 |
| data/news\_client.py                  |       57 |        2 |     96% |     88-89 |
| data/noaa\_client.py                  |      129 |        6 |     95% |212-215, 230-231 |
| data/preprocessing.py                 |      120 |        2 |     98% |   288-289 |
| data/quality.py                       |       60 |        4 |     93% |90-91, 126, 130 |
| data/redis\_client.py                 |       84 |        9 |     89% |   196-204 |
| data/session\_diff.py                 |      152 |        5 |     97% |98, 120, 125, 179-180 |
| data/user\_prefs.py                   |      127 |       28 |     78% |94-100, 183, 188-192, 210, 237, 249-261, 263 |
| data/vintage.py                       |      150 |        4 |     97% |115, 181, 237, 382 |
| data/weather\_aggregate.py            |       44 |        3 |     93% |63, 78, 109 |
| data/weather\_client.py               |      255 |       22 |     91% |119-120, 174-188, 300, 303-305, 504-506 |
| data/weather\_normals.py              |      120 |        7 |     94% |94, 130-132, 257, 259-260 |
| models/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| models/arima\_model.py                |      182 |       28 |     85% |95-99, 123-127, 163-211, 329-331, 400, 402 |
| models/benchmark.py                   |      176 |        9 |     95% |172, 191, 365, 370-371, 409, 412-413, 667 |
| models/drift.py                       |      277 |       17 |     94% |145, 160, 203, 206, 315, 358, 360, 374, 448-449, 603, 650-651, 656, 689, 710-711 |
| models/ensemble.py                    |       46 |        3 |     93% |103-104, 119 |
| models/evaluation.py                  |       92 |        3 |     97% |156, 184, 206 |
| models/model\_service.py              |      316 |       14 |     96% |148-149, 169-170, 241, 544-549, 693, 697-699 |
| models/persistence.py                 |      306 |       53 |     83% |42, 88-101, 174-175, 200-202, 210-211, 319, 339, 355-372, 384-385, 459-467, 540-548, 568, 570-577, 604-610, 625-626, 638-644, 646-654, 668-676, 695 |
| models/pricing.py                     |       32 |        0 |    100% |           |
| models/prophet\_model.py              |       74 |        4 |     95% |208, 231-238 |
| models/rolling\_eval.py               |       97 |        2 |     98% |  159, 174 |
| models/skill.py                       |       52 |        2 |     96% |   41, 179 |
| models/training.py                    |      100 |        2 |     98% |   184-185 |
| models/xgboost\_model.py              |       73 |        2 |     97% |    75, 99 |
| personas/\_\_init\_\_.py              |        0 |        0 |    100% |           |
| personas/config.py                    |       13 |        0 |    100% |           |
| personas/welcome.py                   |       90 |        4 |     96% | 48-51, 97 |
| simulation/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| simulation/presets.py                 |        7 |        0 |    100% |           |
| simulation/scenario\_engine.py        |       65 |        8 |     88% |187-193, 200-202 |
| **TOTAL**                             | **8531** |  **951** | **89%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/kristenmartino/gridpulse/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kristenmartino/gridpulse/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fkristenmartino%2Fgridpulse%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.