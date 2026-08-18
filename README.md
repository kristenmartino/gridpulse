# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                  |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------- | -------: | -------: | ------: | --------: |
| components/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| components/\_callbacks\_alerts.py     |      263 |       51 |     81% |430, 438-439, 515-520, 537-539, 570, 644-645, 688-689, 718-769 |
| components/\_callbacks\_backtest.py   |      285 |       44 |     85% |95, 103, 105, 139-156, 174-177, 212, 223, 241-242, 567-572, 580-581, 640-643, 685-687, 697-699, 780-781 |
| components/\_callbacks\_forecast.py   |      876 |      160 |     82% |376, 439-440, 542, 555-558, 573-576, 605-608, 621-626, 645-650, 670-675, 711-712, 728, 756-757, 768, 777-789, 874, 1211, 1226-1228, 1352-1354, 1371-1373, 1387-1399, 1445-1447, 1458-1463, 1486, 1490-1491, 1497, 1503-1505, 1577, 1605-1609, 1705-1706, 1742-1743, 1847-1865, 1889-1942, 1966, 1975, 2045-2135, 2139-2142, 2146, 2194, 2204, 2241, 2366, 2450-2451, 2460 |
| components/\_callbacks\_generation.py |       72 |        4 |     94% |74, 99-100, 112 |
| components/\_callbacks\_models.py     |      382 |       17 |     96% |94, 179, 633, 684, 835, 943, 962, 980-983, 1023, 1085-1086, 1228, 1250-1259, 1273 |
| components/\_callbacks\_overview.py   |      306 |       16 |     95% |396, 403, 411, 524, 574, 582, 584, 616, 722, 785, 888, 935-942 |
| components/\_callbacks\_shared.py     |      265 |       21 |     92% |384-392, 409-410, 622-623, 630-631, 746, 748, 844, 869-871 |
| components/\_callbacks\_us\_grid.py   |      302 |       46 |     85% |448, 634, 797, 926-982, 994-1002, 1056, 1059 |
| components/\_callbacks\_weather.py    |       42 |        0 |    100% |           |
| components/accessibility.py           |       26 |        3 |     88% |169, 186-187 |
| components/callbacks.py               |      360 |       84 |     77% |141-143, 221, 361-366, 514-562, 738, 778-779, 813-841, 857-871, 888-901, 910-922, 949-950 |
| components/cards.py                   |       88 |        5 |     94% |139-142, 172 |
| components/error\_handling.py         |      106 |       13 |     88% |52, 99, 248-253, 315, 366-370, 454, 469 |
| components/icons.py                   |       14 |        1 |     93% |       123 |
| components/insights.py                |      351 |        1 |     99% |       483 |
| components/layout.py                  |       29 |        1 |     97% |        78 |
| components/tab\_alerts.py             |       15 |        0 |    100% |           |
| components/tab\_demand\_outlook.py    |       42 |        0 |    100% |           |
| components/tab\_models.py             |       15 |        0 |    100% |           |
| components/tab\_overview.py           |        4 |        0 |    100% |           |
| components/tab\_us\_grid.py           |        9 |        0 |    100% |           |
| data/\_\_init\_\_.py                  |        7 |        0 |    100% |           |
| data/audit.py                         |       54 |        0 |    100% |           |
| data/cache.py                         |      102 |        2 |     98% |  193, 205 |
| data/demo\_data.py                    |       89 |        0 |    100% |           |
| data/eia\_client.py                   |      248 |        2 |     99% |  380, 382 |
| data/explainability.py                |       44 |        3 |     93% |   204-206 |
| data/feature\_engineering.py          |      236 |        8 |     97% |59-60, 184, 514, 713-714, 872, 877 |
| data/forecast\_history.py             |      137 |       11 |     92% |54-55, 116-117, 149, 157, 181-183, 420-421 |
| data/gcs\_store.py                    |       71 |        3 |     96% |31, 52, 99 |
| data/noaa\_client.py                  |      129 |        6 |     95% |212-215, 230-231 |
| data/preprocessing.py                 |      120 |        2 |     98% |   288-289 |
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
| models/benchmark.py                   |      201 |        9 |     96% |172, 191, 430, 435-436, 474, 477-478, 732 |
| models/drift.py                       |      341 |       15 |     96% |220, 235, 278, 281, 464, 560, 686-687, 950, 1005-1006, 1012, 1056, 1078-1079 |
| models/ensemble.py                    |       90 |        1 |     99% |        91 |
| models/evaluation.py                  |       92 |        0 |    100% |           |
| models/model\_service.py              |      319 |       14 |     96% |148-149, 169-170, 241, 572-577, 721, 725-727 |
| models/persistence.py                 |      306 |       53 |     83% |42, 88-101, 174-175, 200-202, 210-211, 319, 339, 355-372, 384-385, 459-467, 540-548, 568, 570-577, 604-610, 625-626, 638-644, 646-654, 668-676, 695 |
| models/pricing.py                     |       32 |        0 |    100% |           |
| models/prophet\_model.py              |       74 |        4 |     95% |208, 231-238 |
| models/rolling\_eval.py               |       97 |        0 |    100% |           |
| models/shadow\_eval.py                |       63 |        5 |     92% |80-81, 164-165, 200 |
| models/skill.py                       |       52 |        0 |    100% |           |
| models/training.py                    |      100 |        2 |     98% |   184-185 |
| models/xgboost\_model.py              |       84 |        7 |     92% |83, 107, 127-131 |
| personas/\_\_init\_\_.py              |        0 |        0 |    100% |           |
| personas/config.py                    |       13 |        0 |    100% |           |
| personas/welcome.py                   |       90 |        4 |     96% | 48-51, 97 |
| simulation/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| simulation/presets.py                 |        7 |        0 |    100% |           |
| simulation/scenario\_engine.py        |       78 |        8 |     90% |258-264, 271-273 |
| simulation/scenario\_grid.py          |      104 |        9 |     91% |121, 234, 259, 261, 305-307, 310, 314 |
| **TOTAL**                             | **8496** |  **735** | **91%** |           |


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