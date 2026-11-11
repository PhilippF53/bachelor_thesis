# NL2SQL-Benchmark auf MIMICSQL mit Open-Source-LLMs

Benchmark lokaler LLMs (über Ollama) auf MIMICSQL (MimicSQL2, template-basierte Fragen)
mit [NL2SQL360](https://github.com/HKUSTDial/NL2SQL360) als Evaluationsrahmen:
Execution Accuracy (EX), Exact Match (EM), Soft F1, VES und R-VES, Latenz, Tokens und
Fehleranalyse.

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
- **`evaluate`** schreibt `per_sample.csv` jedes Mal neu. Die VES-Messung (siehe
  [Methodische Entscheidungen](#methodische-entscheidungen)) dauert rund eine Minute je
  Lauf und wird in `efficiency.csv` zwischengespeichert. Ein erneutes `evaluate` misst nur
  Vorhersagen neu, deren SQL sich geändert hat. Weil Laufzeiten gemessen werden, sollte
  `evaluate` laufen, wenn der Rechner sonst nichts tut (nicht parallel zu `infer`). Eine bestehende
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
| `runs/<split>/<model>/<variant>/` | `manifest.json` (Config-Snapshot, Einstellungen, Zeitstempel, Tag und Digest, Ollama-Version, Git-Commit, GPU-Platzierung und geladene Kontextlänge), `raw.jsonl` (Prompt, Rohausgabe, Denk-Ausgabe getrennt, extrahierte SQL, Timings, Tokens), `predictions.sql` |
| `results/<split>/` | je Split (`train`, `dev`, `test`): `gold_check.csv`; für dev und test zusätzlich `runs/<model>/<variant>/per_sample.csv` und `efficiency.csv` (VES-Cache), `summary.csv/json`, `all_samples.csv`, Fehlertabellen, `solved_by_models.csv`, `figures/` (Liste der Diagramme in `docs/architektur.md`). train enthält nur die Gold-Prüfung, weil darauf keine Modelle laufen (Beispielpool für Few-Shot). |
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
- **Soft F1** (BIRD mini-dev, `F1Evaluator` von NL2SQL360) gibt Teilpunkte, wenn das
  Ergebnis teilweise stimmt: Zellen werden zeilenweise verglichen, Reihenfolge und doppelte
  Zeilen zählen nicht. Berechnet wird F1 aus den Ergebnissen, die für EX ohnehin im
  Speicher liegen. Eine Abweichung von BIRD: Dort werden die Zeilen in der Reihenfolge
  eines Python-`set` gepaart, die sich bei Strings von Prozess zu Prozess ändert. Hier
  werden sie vorher stabil sortiert, sonst wäre F1 nicht reproduzierbar. 83 % der
  dev-Gold-Ergebnisse haben genau eine Zeile; dort ist F1 fast immer gleich EX. F1
  unterscheidet sich vor allem bei Listen-Fragen und knapp falschen Antworten
  (Diagramm `f1_wrong_answers`).
- **VES und R-VES** (BIRD, `VesEvaluator`/`RVesEvaluator` von NL2SQL360) gewichten
  richtige Antworten mit dem Laufzeitverhältnis Gold/Vorhersage: VES mit dessen Wurzel,
  R-VES mit Belohnungsstufen (≥ 2 → 1,25, ≥ 1 → 1, ≥ 0,5 → 0,75, ≥ 0,25 → 0,5, sonst
  0,25). Gemessen wird wie in NL2SQL360 (100 Wiederholungen, 3σ-Ausreißerfilter), aber
  sequenziell, read-only und mit einer gemeinsamen Messung für beide Metriken.
  **Einschränkung:** Auf der MIMICSQL-DB (100 Aufnahmen) laufen die Gold-Queries im Median
  0,1 ms. Die gemessenen Verhältnisse liegen deshalb fast alle sehr nah bei 1 (z.B.
  DuckDB-NSQL, zero-shot: Median 1,00, 90 % zwischen 0,99 und 1,05). VES liegt damit
  praktisch auf Höhe der EX. R-VES ist zusätzlich verzerrt: Bei gleich schnellen Queries
  landet etwa die Hälfte zufällig knapp unter 1 und bekommt 0,75 statt 1. Beide Werte
  sind der Vollständigkeit halber angegeben; Aussagen über die Effizienz der Queries
  tragen sie auf dieser kleinen DB kaum.
- **NL2SQL360:** Die Ergebnisse stehen zusätzlich in den NL2SQL360-Tabellen
  (`data/derived/nl2sql360/nl2sql360.sqlite`), mit EX, EM, F1, VES und R-VES.
  Leaderboards und Filter funktionieren dort also wie gewohnt.
- **Prompts:** Alle Modelle bekommen denselben Inhalt: Schema mit Spaltenbeschreibungen,
  Anweisung mit ausdrücklichem SQLite-Hinweis, k Beispiele aus train. SQLCoder und
  DuckDB-NSQL erhalten ihn als Rohtext ohne Chat-Template (raw): DuckDB-NSQL im Layout
  seiner Model Card, SQLCoder im Layout aus dem nächsten Punkt. Gesetzt sind
  `temperature=0`, `seed=42` und `num_ctx` explizit.
- **SQLCoder-Layout:** Der Prompt folgt Defogs aktuellem Layout mit `[QUESTION]…[/QUESTION]`
  und `[SQL]`, das für `sqlcoder-7b-2` dokumentiert ist. Die Ollama-Tags `sqlcoder:7b` und
  `sqlcoder:15b` sind ältere Checkpoints, für die Defog andere Layouts dokumentiert hat
  (der Prompt endet dort mit einem geöffneten SQL-Codeblock). Ein Check mit diesen beiden
  Layouts auf den ersten 100 dev-Fragen brachte keinen Vorteil. EX in % (bisheriges Layout /
  Layout von `defog/sqlcoder` / Layout von `defog/sqlcoder-7b` und `sqlcoder2`):
  sqlcoder-7b 3 / 4 / 3 mit `zero_shot` und 10 / 17 / 9 mit `few_similar_k3_values`,
  sqlcoder-15b 49 / 52 / 41 und 78 / 49 / 47. Deshalb bleibt es beim bisherigen Layout.
  Code und Config des Checks stehen in Commit `cc07b63`, die Läufe unter
  `runs/_archive/20261111_format_check_sqlcoder/`.
- **Kontextlänge:** Passt ein Prompt samt `num_predict` nicht in `num_ctx`, bricht der
  Lauf ab. Ollama kürzt ein `num_ctx` oberhalb der Kontextlänge des Modells ohne
  Fehlermeldung. `infer` vergleicht deshalb nach dem Laden die tatsächlich geladene
  Kontextlänge (`ollama ps`) mit `num_ctx` und bricht ab, wenn sie kleiner ist. Das
  betrifft `codellama:70b-instruct` mit 2048 Tokens; dort gelten `num_ctx: 2048` und
  `num_predict: 400`. Die dev-Läufe dieses Modells liefen noch mit angeforderten 8192,
  also effektiv 2048 Tokens, und `num_predict: 1024`. Von den 3184 Antworten erreichte eine
  die Kontextgrenze (`few_similar_k3_values`, 1024 Ausgabe-Tokens, fünfmal derselbe
  SQL-Block); die extrahierte SQL entspricht dem ersten Block. Drei Antworten waren länger
  als 400 Tokens, bei allen steht der vollständige SQL-Block in den ersten rund 300 Tokens.
- **Varianten:** zero-shot, few-shot k=3 zufällig (Seed pro Frage, für alle Modelle
  gleich), few-shot k=3 ähnlichkeitsbasiert (TF-IDF) und dieselbe Variante mit
  Wert-Hinweisen (`few_similar_k3_values`). Verglichen wird auf dev, test nur für den
  finalen Lauf.
- **Wert-Hinweise:** Die Template-Fragen sind kleingeschrieben, die DB speichert Werte in
  ihrer Originalschreibweise, und `=` auf Text unterscheidet Groß-/Kleinschreibung. Bei 874
  von 1.376 Textwerten der dev-Gold-SQL steht die DB-Schreibweise nicht in der Frage. Die
  Variante `few_similar_k3_values` listet unter dem Schema die DB-Werte der Textspalten, die
  (ohne Groß-/Kleinschreibung, an Wortgrenzen) wörtlich in der Frage vorkommen, in
  DB-Schreibweise, z.B. `-- DEMOGRAPHIC.GENDER = 'F'`. Zeit- und Messwertspalten und reine
  Zahlen sind ausgenommen. Auf dev sind das im Mittel 1,5 Werte bzw. ca. 130 Zeichen pro
  Prompt; die Werte decken alle 874 Fälle ab. Die anderen Varianten bleiben unverändert.
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
  dieses Modells sind deshalb nicht mit denen der anderen Modelle vergleichbar. Im
  Diagramm „von wie vielen Modellen gelöst“ fehlt es (`report.contaminated_models`).
- **Verweigerungen bei CodeLlama-70b:** `codellama:70b-instruct` beantwortet einen Teil der
  Fragen nicht mit SQL, sondern lehnt sie ab oder bittet um Klärung, meist mit „The question
  is not clear“ oder „The question is not specific enough“. Diese Antworten enthalten keine
  SQL. Sie zählen als falsch (EX = 0) und stehen in `summary.csv` in der Spalte `no_sql`
  sowie in der Fehleranalyse in der Kategorie `no_sql`; die Antworttexte stehen in
  `raw.jsonl` (`content`). Das ist Verhalten des Modells, kein Fehler der SQL-Extraktion.
  Die kleineren CodeLlama-Modelle (7b, 13b, 34b) zeigen es in keinem dev-Lauf.
- **Datenschutz:** Abfrageergebnisse werden nur im Speicher verglichen. Gespeichert
  werden korrekt/falsch, Soft-F1-Wert, Zeilenanzahl, Fehlermeldung und Laufzeiten.
