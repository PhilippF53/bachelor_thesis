# Gold-SQL-Prüfung (MIMICSQL auf nachgebauter TREQS-DB)

Datenbank: `data/derived/database/mimicsql/mimicsql.sqlite`, Timeout 30s.

| split   |    n |   exec_ok |   empty_result |   sql_norm_equal |   sqlglot_error |   spider_parse_error |   median_runtime_s |
|:--------|-----:|----------:|---------------:|-----------------:|----------------:|---------------------:|-------------------:|
| train   | 8346 |      8346 |             10 |             8346 |               0 |                   30 |             0.0001 |
| dev     |  796 |       796 |              4 |              796 |               0 |                    0 |             0.0001 |
| test    | 1000 |      1000 |              4 |             1000 |               0 |                    0 |             0.0001 |

`empty_result` = Gold liefert 0 Zeilen (Plausibilitätstest für den DB-Nachbau). `sql_norm_equal` = Variante mit einfachen Anführungszeichen (für Few-Shot-Beispiele) liefert dasselbe Ergebnis.

## Ausführungsfehler (0)


## Spider-Parser (Exact Match) kann Gold nicht verarbeiten (30)

| Fehler                           |   Anzahl |
|:---------------------------------|---------:|
| AssertionError: Unexpected quote |       30 |

## Verteilung Schwierigkeit

| spider_hardness   |   train |   dev |   test |
|:------------------|--------:|------:|-------:|
| easy              |     686 |    61 |     69 |
| medium            |    6719 |   696 |    868 |
| hard              |     911 |    39 |     63 |
| parse_error       |      30 |     0 |      0 |
