# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/kristenmartino/gridpulse/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                  |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------- | -------: | -------: | ------: | --------: |
| components/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| components/\_callbacks\_alerts.py     |      198 |       17 |     91% |429-431, 439-440, 517-522, 539-541, 572, 646-647, 690-691 |
| components/\_callbacks\_backtest.py   |      285 |       44 |     85% |95, 103, 105, 139-156, 174-177, 212, 223, 241-242, 567-572, 580-581, 640-643, 685-687, 697-699, 780-781 |
| components/\_callbacks\_forecast.py   |      656 |      123 |     81% |366, 429-430, 532, 545-548, 563-566, 595-598, 611-616, 635-640, 660-665, 701-702, 718, 746-747, 758, 767-779, 864, 1201, 1216-1218, 1348-1350, 1367-1369, 1383-1395, 1441-1443, 1454-1459, 1482, 1486-1487, 1493, 1499-1501, 1573, 1601-1605, 1701-1702, 1738-1739, 1843-1861, 1885-1938 |
| components/\_callbacks\_generation.py |       72 |        4 |     94% |74, 99-100, 112 |
| components/\_callbacks\_models.py     |      350 |       13 |     96% |93, 178, 632, 683, 834, 943, 962, 980-983, 1023, 1085-1086 |
| components/\_callbacks\_overview.py   |      938 |      206 |     78% |154, 179-181, 276-278, 479, 486, 494, 607, 657, 661, 665, 667, 676, 689-699, 755, 787, 850, 916, 925, 995-1085, 1089-1092, 1096, 1144, 1154, 1191, 1316, 1338, 1360-1369, 1383, 1399-1450, 1531-1532, 1541, 1804, 1845-1904, 2029-2074, 2108, 2144, 2181, 2183, 2191, 2258, 2481-2482, 2489-2494, 2502-2513, 2516-2518, 2523-2555, 2661-2665, 2859-2898 |
| components/\_callbacks\_shared.py     |      235 |       19 |     92% |380-388, 405-406, 411-412, 618-619, 626-627, 742, 744 |
| components/\_callbacks\_us\_grid.py   |      302 |       46 |     85% |448, 634, 797, 926-982, 994-1002, 1056, 1059 |
| components/\_callbacks\_weather.py    |       42 |        0 |    100% |           |
| components/accessibility.py           |       26 |        3 |     88% |169, 186-187 |
| components/callbacks.py               |      360 |       84 |     77% |151-153, 231, 371-376, 524-572, 748, 788-789, 823-851, 867-881, 898-911, 920-932, 959-960 |
| components/cards.py                   |       97 |        6 |     94% |193-196, 226, 369 |
| components/error\_handling.py         |      106 |       13 |     88% |52, 99, 248-253, 315, 366-370, 454, 469 |
| components/icons.py                   |       14 |        1 |     93% |       123 |
| components/insights.py                |      351 |        1 |     99% |       468 |
| components/layout.py                  |       29 |        1 |     97% |        78 |
| components/tab\_alerts.py             |       15 |        0 |    100% |           |
| components/tab\_demand\_outlook.py    |       42 |        0 |    100% |           |
| components/tab\_models.py             |       15 |        0 |    100% |           |
| components/tab\_overview.py           |        4 |        0 |    100% |           |
| components/tab\_us\_grid.py           |        9 |        0 |    100% |           |
| data/\_\_init\_\_.py                  |        7 |        0 |    100% |           |
| data/ai\_briefing.py                  |      193 |       82 |     58% |77-81, 100-135, 204, 213, 216-217, 228-232, 237-242, 245, 258-269, 274-275, 280, 293-295, 300, 303, 350-364, 369-379, 386-410, 423 |
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
| data/vintage.py                       |      150 |        4 |     97% |115, 181, 237, 382 |
| data/weather\_aggregate.py            |       44 |        3 |     93% |63, 78, 109 |
| data/weather\_client.py               |      307 |       19 |     94% |120-121, 175-189, 301, 676-678 |
| data/weather\_normals.py              |      120 |        7 |     94% |94, 130-132, 257, 259-260 |
| models/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| models/arima\_model.py                |      187 |       28 |     85% |95-99, 123-127, 165-218, 371-373, 442, 444 |
| models/benchmark.py                   |      176 |        9 |     95% |172, 191, 365, 370-371, 409, 412-413, 667 |
| models/drift.py                       |      312 |       15 |     95% |174, 189, 232, 235, 400, 455, 568-569, 759, 806-807, 812, 845, 866-867 |
| models/ensemble.py                    |       84 |        1 |     99% |        91 |
| models/evaluation.py                  |       92 |        0 |    100% |           |
| models/model\_service.py              |      319 |       14 |     96% |148-149, 169-170, 241, 572-577, 721, 725-727 |
| models/persistence.py                 |      306 |       53 |     83% |42, 88-101, 174-175, 200-202, 210-211, 319, 339, 355-372, 384-385, 459-467, 540-548, 568, 570-577, 604-610, 625-626, 638-644, 646-654, 668-676, 695 |
| models/pricing.py                     |       32 |        0 |    100% |           |
| models/prophet\_model.py              |       74 |        4 |     95% |208, 231-238 |
| models/rolling\_eval.py               |       97 |        0 |    100% |           |
| models/shadow\_eval.py                |       20 |        1 |     95% |        36 |
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
| **TOTAL**                             | **8864** |  **925** | **90%** |           |


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