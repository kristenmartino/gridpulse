# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                  |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------- | -------: | -------: | ------: | --------: |
| components/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| components/\_callbacks\_alerts.py     |      251 |       16 |     94% |420, 428-429, 505-510, 527-529, 560, 632-633, 676-677 |
| components/\_callbacks\_backtest.py   |      285 |       44 |     85% |95, 103, 105, 139-156, 174-177, 212, 223, 241-242, 568-573, 581-582, 641-644, 686-688, 698-700, 781-782 |
| components/\_callbacks\_forecast.py   |      849 |      155 |     82% |379, 442-443, 545, 558-561, 576-579, 608-611, 624-629, 648-653, 673-678, 714-715, 731, 759-760, 771, 780-792, 877, 1100-1102, 1119-1120, 1208, 1318-1320, 1337-1339, 1353-1365, 1411-1413, 1424-1429, 1503, 1531-1535, 1639-1640, 1676-1677, 1781-1799, 1823-1876, 1900, 1909, 1979-2069, 2073-2076, 2080, 2128, 2138, 2175, 2300, 2384-2385, 2394 |
| components/\_callbacks\_generation.py |       72 |        4 |     94% |74, 99-100, 112 |
| components/\_callbacks\_models.py     |      405 |       15 |     96% |94, 179, 596, 713, 885, 982, 1001, 1019-1022, 1062, 1126-1127, 1265, 1312 |
| components/\_callbacks\_overview.py   |      288 |        7 |     98% |398, 405, 413, 511, 648, 658, 751 |
| components/\_callbacks\_shared.py     |      283 |       26 |     91% |384-392, 409-410, 415-416, 622-623, 630-631, 746, 748, 842, 867-869, 892, 894-895 |
| components/\_callbacks\_us\_grid.py   |      302 |       46 |     85% |449, 636, 799, 928-984, 996-1004, 1058, 1061 |
| components/\_callbacks\_weather.py    |       42 |        0 |    100% |           |
| components/accessibility.py           |       26 |        3 |     88% |169, 186-187 |
| components/callbacks.py               |      360 |       84 |     77% |141-143, 221, 361-366, 514-562, 738, 778-779, 813-841, 857-871, 888-901, 910-922, 949-950 |
| components/cards.py                   |       88 |       13 |     85% |139-142, 172, 288-325 |
| components/error\_handling.py         |      106 |       13 |     88% |52, 99, 238-243, 305, 354-358, 442, 457 |
| components/forecast\_context.py       |       72 |        0 |    100% |           |
| components/forecast\_review.py        |        6 |        0 |    100% |           |
| components/icons.py                   |       14 |        1 |     93% |       123 |
| components/insights.py                |      349 |        1 |     99% |       483 |
| components/layout.py                  |       29 |        1 |     97% |        76 |
| components/tab\_alerts.py             |       15 |        0 |    100% |           |
| components/tab\_demand\_outlook.py    |       42 |        0 |    100% |           |
| components/tab\_models.py             |       15 |        0 |    100% |           |
| components/tab\_overview.py           |        4 |        0 |    100% |           |
| components/tab\_us\_grid.py           |        9 |        0 |    100% |           |
| data/\_\_init\_\_.py                  |        7 |        0 |    100% |           |
| data/audit.py                         |       54 |        0 |    100% |           |
| data/cache.py                         |      102 |        2 |     98% |  198, 210 |
| data/demo\_data.py                    |       89 |        0 |    100% |           |
| data/eia\_client.py                   |      248 |        2 |     99% |  380, 382 |
| data/explainability.py                |       44 |        3 |     93% |   205-207 |
| data/feature\_engineering.py          |      407 |       30 |     93% |61-62, 189, 338-339, 357-359, 443, 493, 611, 617, 638, 641-642, 661, 853-862, 866, 898, 916, 972, 1171-1172, 1330, 1335 |
| data/forecast\_evidence.py            |       31 |        0 |    100% |           |
| data/forecast\_history.py             |      137 |       11 |     92% |54-55, 116-117, 149, 157, 181-183, 420-421 |
| data/forecast\_identity.py            |       63 |        1 |     98% |        49 |
| data/forecast\_publications.py        |      114 |        5 |     96% |41, 108, 117, 137, 229 |
| data/gcs\_store.py                    |       71 |        3 |     96% |31, 52, 99 |
| data/hourly\_scoring\_store.py        |       29 |        2 |     93% |    28, 47 |
| data/noaa\_client.py                  |      129 |        6 |     95% |212-215, 230-231 |
| data/preprocessing.py                 |      128 |        2 |     98% |   313-314 |
| data/quality.py                       |       60 |        4 |     93% |90-91, 126, 130 |
| data/redis\_client.py                 |      101 |        0 |    100% |           |
| data/session\_diff.py                 |      152 |        5 |     97% |98, 120, 125, 179-180 |
| data/user\_prefs.py                   |      127 |       28 |     78% |94-100, 183, 188-192, 210, 237, 249-261, 263 |
| data/vintage.py                       |      166 |        6 |     96% |123, 199, 302, 364-365, 501 |
| data/weather\_aggregate.py            |       44 |        3 |     93% |63, 78, 109 |
| data/weather\_client.py               |      307 |       19 |     94% |120-121, 175-189, 301, 676-678 |
| data/weather\_normals.py              |      120 |        7 |     94% |94, 130-132, 257, 259-260 |
| models/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| models/arima\_model.py                |      187 |       28 |     85% |95-99, 123-127, 165-218, 371-373, 442, 444 |
| models/benchmark.py                   |      245 |        9 |     96% |209, 228, 386, 709, 714-715, 753, 756-757 |
| models/drift.py                       |      396 |       19 |     95% |220, 235, 280, 283, 466, 562, 688-689, 952, 1014-1015, 1021, 1065, 1096-1097, 1129-1130, 1136-1137 |
| models/ensemble.py                    |       90 |        1 |     99% |        91 |
| models/evaluation.py                  |       92 |        0 |    100% |           |
| models/hourly\_scoring.py             |      126 |        6 |     95% |51, 66, 127, 178, 180, 193 |
| models/model\_service.py              |      319 |       14 |     96% |148-149, 169-170, 241, 572-577, 721, 725-727 |
| models/persistence.py                 |      306 |       53 |     83% |42, 88-101, 174-175, 200-202, 210-211, 319, 339, 355-372, 384-385, 459-467, 540-548, 568, 570-577, 604-610, 625-626, 638-644, 646-654, 668-676, 695 |
| models/pricing.py                     |       32 |        0 |    100% |           |
| models/prophet\_model.py              |       92 |        4 |     96% |265, 288-295 |
| models/rolling\_eval.py               |       97 |        0 |    100% |           |
| models/shadow\_eval.py                |       63 |        5 |     92% |80-81, 164-165, 200 |
| models/skill.py                       |       52 |        0 |    100% |           |
| models/training.py                    |      100 |        2 |     98% |   188-189 |
| models/xgboost\_model.py              |       84 |        7 |     92% |83, 107, 127-131 |
| personas/\_\_init\_\_.py              |        0 |        0 |    100% |           |
| personas/config.py                    |       13 |        0 |    100% |           |
| personas/welcome.py                   |       90 |        3 |     97% |47, 49, 97 |
| simulation/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| simulation/presets.py                 |        7 |        0 |    100% |           |
| simulation/scenario\_engine.py        |       78 |        8 |     90% |258-264, 271-273 |
| simulation/scenario\_grid.py          |      104 |        9 |     91% |121, 234, 259, 261, 305-307, 310, 314 |
| **TOTAL**                             | **9215** |  **736** | **92%** |           |


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