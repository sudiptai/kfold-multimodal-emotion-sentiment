| modality   | task      | model   | K-fold val acc (mean ± std)   |   held-out test acc |   n_test |
|:-----------|:----------|:--------|:------------------------------|--------------------:|---------:|
| audio      | emotion   | CNN     | 0.874 ± 0.012                 |               0.849 |      826 |
| audio      | emotion   | LSTM    | 0.871 ± 0.009                 |               0.878 |      826 |
| audio      | sentiment | CNN     | 0.771 ± 0.013                 |               0.736 |      734 |
| audio      | sentiment | LSTM    | 0.773 ± 0.011                 |               0.753 |      734 |