# NL2SQL-Benchmark auf MIMICSQL mit Open-Source-LLMs

Benchmark lokaler LLMs (über Ollama) auf MIMICSQL (MimicSQL2, template-basierte Fragen)
mit [NL2SQL360](https://github.com/HKUSTDial/NL2SQL360) als Evaluationsrahmen:
Execution Accuracy (EX), Exact Match (EM), Latenz, Tokens und Fehleranalyse.

## Voraussetzungen

- Python 3.13 und [uv](https://docs.astral.sh/uv/): `uv sync`
- NLTK-Tokenizer für den Spider-Parser (EM):
  `uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab`
- Ollama ≥ 0.40 mit den Modellen aus `config/config.yaml` (siehe [Modelle](#modelle))
- MIMIC-III als SQLite unter `data/mimic.db` (PhysioNet-Zugang nötig)
- MimicSQL2 unter `data/mimicsql/MimicSQL2` und das TREQS-Repository unter `data/mimicsql/TREQS`

## Reproduzieren

Alles wird über `config/config.yaml` gesteuert (Pfade, Modelle, Prompt-Varianten, Seeds).

```bash
uv run python -m src.cli build-db      # MIMICSQL-DB nach TREQS nachbauen (100 Aufnahmen)
uv run python -m src.cli prepare       # MimicSQL2 -> NL2SQL360-Format, tables.json, Import
uv run python -m src.cli check-gold    # Gold-SQL ausführen/parsen -> docs/gold_check.md
uv run python -m src.cli infer --split dev            # alle Modelle x Prompt-Varianten auf dev
uv run python -m src.cli evaluate --split dev
# selection.test_variant in der Config setzen (beste Variante auf dev), dann:
uv run python -m src.cli infer --split test
uv run python -m src.cli evaluate --split test
uv run python -m src.cli report        # Fehleranalyse, Tabellen, Diagramme
```

`infer`, `evaluate` und `reextract` nehmen optional `--models <name …>` und
`--variants <name …>` (Namen aus der Config); `--split` ist `dev` oder `test`.
Testlauf mit den ersten 10 Fragen: `infer --split dev --limit 10`.

Unit-Tests: `uv run pytest`

### Fortschrittsanzeige

`infer` zeigt einen Gesamtbalken über alle offenen Antworten (mit ETA) und darunter einen
Balken für die aktuelle Kombination aus Modell und Variante, z.B.
`[3/5 | 2/3] gemma4-31b/few_similar_k3`. Dazu kommen laufende mittlere Latenz,
Output-Tokens und die Zahl der Antworten ohne SQL. Log-Zeilen melden Laden, Warm-up
(Ladezeit, längster Prompt gegen `num_ctx`, GPU-Platzierung), Abschluss und Entladen
jedes Modells.

### Fortsetzen und erneute Läufe

- **`infer`** setzt einen abgebrochenen Lauf an der letzten beantworteten Frage fort.
  Bestehende Antworten werden nie überschrieben. Ein Testlauf mit `--limit 10` geht
  so nahtlos in den vollen Lauf über.
- Vor dem Laden des Modells prüft `infer`, ob sich etwas geändert hat: Modell, Optionen,
  Variante, Schema, Prompt-Vorlage (Hash) oder Modell-Digest. Außerdem muss die aktuelle
  Fassung jeden gespeicherten **Prompt-Text** exakt reproduzieren, also auch Fragen und
  Few-Shot-Beispiele. Bei einer Abweichung bricht der Lauf ab, statt Antworten auf
  verschiedene Prompts zu mischen. Dann den alten Ordner unter `runs/` archivieren oder löschen.
- **`reextract`** extrahiert die SQL neu aus den gespeicherten Rohausgaben, ohne neue
  Inferenz (z.B. nach einer Änderung an `src/inference/extract.py`). Geändert werden nur
  `pred_sql` und `extract_method` in `raw.jsonl` sowie `predictions.sql`; das Manifest
  protokolliert den Vorgang (`reextracted`). Danach `evaluate --overwrite` und `report`.
  `evaluate` warnt, wenn gespeicherte SQL und aktuelle Extraktion voneinander abweichen.
- **`evaluate`** schreibt `per_sample.csv` jedes Mal neu. Eine bestehende
  NL2SQL360-Bewertungstabelle wird nur mit `--overwrite` ersetzt. Unvollständige Läufe
  (z.B. `--limit`) landen nur in der CSV, nicht in NL2SQL360.
- **`report`** erzeugt alle Tabellen und Diagramme neu, aus allen vorhandenen
  `per_sample.csv`. Ältere Ergebnisse, die erhalten bleiben sollen, vorher verschieben
  (z.B. nach `results/_archive/<zeitstempel>/`).

## Modelle

| Name in der Config | Ollama-Tag | Größe | Format | Reasoning |
|---|---|---|---|---|
| `duckdb-nsql-7b` | `duckdb-nsql:latest` | 3.8 GB | DuckDB-NSQL (raw) | – |
| `codellama-7b` | `codellama:7b-instruct` | 3.8 GB | Chat | – |
| `sqlcoder-7b` | `sqlcoder:7b` | 4.1 GB | SQLCoder (raw) | – |
| `codellama-13b` | `codellama:13b-instruct` | 7.4 GB | Chat | – |
| `sqlcoder-15b` | `sqlcoder:15b` | 9.0 GB | SQLCoder (raw) | – |
| `gpt-oss-20b` | `gpt-oss:20b` | 13.8 GB | Chat | `medium` |
| `qwen3-coder-30b` | `qwen3-coder:30b` (30.5B MoE) | 18 GB | Chat | – |
| `codellama-34b` | `codellama:34b-instruct` | 19.1 GB | Chat | – |
| `gemma4-31b` | `gemma4:31b` | 20.4 GB | Chat | Thinking an |
| `qwen3.6` | `qwen3.6:latest` (36B MoE) | 23 GB | Chat | Thinking an |
| `codellama-70b` | `codellama:70b-instruct` | 38.9 GB | Chat | – |
| `gpt-oss-120b` | `gpt-oss:120b` | 65.4 GB | Chat | `medium` |
| `llama4-scout` | `llama4:scout` (16x17b) | 67.4 GB | Chat | – |

Die vollen Läufe gehen in dieser Reihenfolge, kleine Modelle zuerst.

Abweichungen von der ursprünglichen Modellliste: Gemma 4 nur als 31b (21b gibt es nicht),
Llama 4 128x17b ausgelassen (245 GB, mehr als die 144 GB VRAM), qwen3-coder:30b und qwen3.6
ergänzt.

Die Denk-Ausgabe von Reasoning-Modellen wird getrennt gespeichert. SQL wird nur aus der
finalen Antwort extrahiert. `output_tokens` enthält auch die Reasoning-Tokens (Ollama zählt
alle erzeugten Tokens).

## Aufbau

| Pfad | Inhalt |
|---|---|
| `config/config.yaml` | Pfade, Modelle, Prompt-Varianten, Seeds, Timeout |
| `src/cli.py` | Einstieg für alle Schritte |
| `src/data/` | DB-Nachbau (TREQS), Datensatz-Konvertierung, Gold-Prüfung |
| `src/inference/` | Prompt-Inhalt, modellspezifische Formate, Few-Shot-Auswahl, SQL-Extraktion, Runner |
| `src/eval/` | EX/EM pro Beispiel, `MimicCore` (NL2SQL360-Erweiterung), Schwierigkeitsgrade |
| `src/analysis/` | Fehlerkategorien, Zusammenfassung, Diagramme |
| `runs/<split>/<model>/<variant>/` | `manifest.json` (Config-Snapshot, Einstellungen, Zeitstempel, Tag und Digest, Ollama-Version, Git-Commit, GPU-Platzierung), `raw.jsonl` (Prompt, Rohausgabe, Denk-Ausgabe getrennt, extrahierte SQL, Timings, Tokens), `predictions.sql` |
| `results/<split>/` | je Split (`train`, `dev`, `test`): `gold_check.csv`; für dev und test zusätzlich `runs/<model>/<variant>/per_sample.csv`, `summary.csv/json`, `all_samples.csv`, Fehlertabellen, `figures/`. train enthält nur die Gold-Prüfung, weil darauf keine Modelle laufen (Beispielpool für Few-Shot). |
| `docs/gold_check.md` | Bericht zur Gold-SQL-Prüfung |
| `data/derived/` | nachgebaute DB, konvertierte Splits, `tables.json`, NL2SQL360-Datenbank |

## Methodische Entscheidungen

- **Datensatz:** MimicSQL2 (Tarbell et al., AMIA 2024) teilt MIMICSQL neu auf (train 8346,
  dev 796, test 1000), um die Generalisierung zu prüfen. `question_refine` entspricht der
  Template-Version der Fragen, nicht den natürlichsprachlichen Paraphrasen. Die Tabelle
  PROCEDURES kommt in train nicht vor, aber in 88 % (dev) bzw. 87 % (test) der Gold-Queries;
  Few-Shot-Beispiele zeigen sie also nie.
- **Duplikate zwischen den Splits:** 86 dev- und 126 test-Fragen stehen mit identischer
  Gold-SQL auch in train (alle mit `FROM PRESCRIPTIONS`). Sie werden bewusst nicht
  gefiltert. `few_similar_k3` wählt für sie das identische Beispiel, also die Lösung. Das
  hebt die EX dieser Variante und vor allem die Stufe easy: 52 der 61 easy-Fragen in dev und
  66 der 69 in test sind Duplikate. `summary.csv` weist deshalb zusätzlich `n_train_dup` und
  `EX_excl_train_dup` (EX ohne diese Fragen) aus, `all_samples.csv` die Spalte
  `train_duplicate`. Die Haupt-EX enthält alle Fragen.
- **Datenbank:** Den Views in `data/mimic.db` fehlen Spalten, die die Gold-SQL nutzt
  (`NAME`, `ADMITTIME`, `DISCHTIME`, `DRUG_DOSE`, `LAB.CHARTTIME`, `VALUE_UNIT`). Deshalb
  wird die originale MIMICSQL-DB mit den unveränderten TREQS-Funktionen nachgebaut:
  100 Aufnahmen, `random_state=0`, synthetische Namen aus `id2name.csv`. `mimic.db` wird
  nur gelesen.
- **Gold-SQL:** Alle 10.142 Gold-Queries (train/dev/test) laufen auf der nachgebauten DB.
  Nur 18 liefern ein leeres Ergebnis. Das spricht dafür, dass dieselben 100 Aufnahmen wie
  im Original gezogen wurden. Details in `docs/gold_check.md`.
- **EX** wird mit eigenem Executor berechnet. Er nutzt `result_eq` aus NL2SQL360, behält
  aber `DISTINCT` (NL2SQL360 entfernt es fest verdrahtet, was bei `COUNT(DISTINCT …)`
  falsch wäre), setzt einen echten Timeout durch und schließt nicht ausführbare
  Gold-Queries aus, statt abzubrechen. Liefert die Gold-SQL ein leeres Ergebnis (4 Fragen in
  dev, 4 in test), zählt jede Vorhersage mit leerem Ergebnis als richtig (Verhalten von
  `result_eq`).
- **EM** nutzt den Spider-Evaluator von NL2SQL360. Vorher werden Gold und Vorhersage
  identisch kanonisiert, weil der Spider-Parser MIMICSQL-Syntax sonst nicht versteht.
  Dabei verlieren Bezeichner aus dem Schema ihre Anführungszeichen (`TABLE."COL"`),
  `INNER JOIN` wird zu `JOIN`, und Spalten-Aliase in SELECT fallen weg. Was danach noch
  nicht parsebar ist, wird gezählt (`em_parse_error_gold/pred`): 30 Gold-Queries in train
  (Apostroph im Wert), keine in dev/test. Antworten ohne SQL zählen als EM = 0, nicht als
  Parse-Fehler. Wie im Spider-Standard ignoriert EM Literalwerte und DISTINCT
  (`DISABLE_VALUE`/`DISABLE_DISTINCT` in NL2SQL360). EM kann deshalb über EX liegen: Bei den
  meisten Modellen sind Werte in falscher Groß-/Kleinschreibung der häufigste Fehler, und den
  erfasst nur EX.
- **NL2SQL360:** Die Ergebnisse stehen zusätzlich in den NL2SQL360-Tabellen
  (`data/derived/nl2sql360/nl2sql360.sqlite`). Leaderboards und Filter funktionieren dort
  also wie gewohnt.
- **Prompts:** Alle Modelle bekommen denselben Inhalt: Schema mit Spaltenbeschreibungen,
  Anweisung mit ausdrücklichem SQLite-Hinweis, k Beispiele aus train. SQLCoder und
  DuckDB-NSQL erhalten ihn in ihrem trainierten Format (raw). Gesetzt sind
  `temperature=0`, `seed=42` und `num_ctx` explizit. Ist ein Prompt zu lang für
  `num_ctx`, bricht der Lauf ab.
- **Varianten:** zero-shot, few-shot k=3 zufällig (Seed pro Frage, für alle Modelle
  gleich) und few-shot k=3 ähnlichkeitsbasiert (TF-IDF). Verglichen wird auf dev, test
  nur für den finalen Lauf.
- **Schwierigkeit:** Spider-Hardness (easy/medium/hard/extra), berechnet mit dem
  Spider-Parser von NL2SQL360 auf der kanonisierten Gold-SQL. Sie steht auch im
  NL2SQL360-Feld `complexity` und ist dort filterbar. dev und test von MimicSQL2
  enthalten keine MIN/MAX/AVG-Queries, nur train. Tabellen und Diagramme ordnen die Stufen
  easy, medium, hard, extra.
- **Fehleranalyse:** Fehler werden regelbasiert kategorisiert (sqlglot-AST und
  Fehlermeldung): Syntax, Dialekt, Tabelle, Spalte, JOIN, Aggregation, DISTINCT,
  Wert-Schreibweise, sonstiger Wert, Operator.
- **Kontamination bei DuckDB-NSQL:** Laut
  [Model Card](https://huggingface.co/motherduckdb/DuckDB-NSQL-7B-v0.1) wurde DuckDB-NSQL
  auch auf Text-to-SQL-Paaren aus NSText2SQL trainiert, und
  [NSText2SQL](https://huggingface.co/datasets/NumbersStation/NSText2SQL) enthält MIMICSQL
  (`mimicsql_data` aus TREQS). Das erklärt die fast perfekten Werte (dev, zero-shot: 99,1 % EX)
  und die richtige Schreibweise der DB-Werte, die nicht im Prompt stehen. Die Ergebnisse
  dieses Modells sind deshalb nicht mit denen der anderen Modelle vergleichbar.
- **Datenschutz:** Abfrageergebnisse werden nur im Speicher verglichen. Gespeichert
  werden korrekt/falsch, Zeilenanzahl, Fehlermeldung und Laufzeit.
