# Architektur und Programmablauf

Dieses Dokument beschreibt, wie der Code intern arbeitet: welche Datei was tut, wie die
Module zusammenspielen und welche Dateien zwischen den Schritten entstehen. Wie die
Pipeline bedient wird und warum methodische Entscheidungen so getroffen wurden, steht in
der [README](../README.md) (Abschnitte *Reproduzieren* und *Methodische Entscheidungen*).

## Überblick

Alle Schritte laufen über einen Einstiegspunkt (`src/cli.py`) und eine
Konfigurationsdatei (`config/config.yaml`):

```
build-db ─▶ prepare ─▶ check-gold ─▶ infer ─▶ evaluate ─▶ report
                                       └──▶ reextract (bei Bedarf, ohne Inferenz)
```

| Schritt | Modul | Liest | Schreibt |
|---|---|---|---|
| `build-db` | `src/data/build_mimicsql_db.py` | `data/mimic.db`, TREQS-Skripte | `data/derived/database/mimicsql/mimicsql.sqlite`, `build_manifest.json` |
| `prepare` | `src/data/convert.py` | MimicSQL2 `new_{split}.json`, nachgebaute DB | `data/derived/mimicsql_{split}.json`, `tables.json`, NL2SQL360-Core-DB |
| `check-gold` | `src/data/gold_check.py` | konvertierte Splits, DB | `results/<split>/gold_check.csv`, `docs/gold_check.md` |
| `infer` | `src/inference/runner.py` | konvertierte Splits, DB-Schema, Ollama | `runs/<split>/<model>/<variant>/` |
| `reextract` | `src/inference/runner.py` | `raw.jsonl` | `pred_sql`/`extract_method` in `raw.jsonl`, `predictions.sql`, Manifest |
| `evaluate` | `src/eval/mimic_core.py` | `raw.jsonl`, Splits, DB | `results/<split>/runs/<model>/<variant>/per_sample.csv` und `efficiency.csv`, NL2SQL360-Evaluationstabellen |
| `report` | `src/analysis/report.py` | alle `per_sample.csv`, Splits | je Split unter `results/<split>/`: `summary.*`, Fehlertabellen, `figures/` |

Abhängigkeiten zwischen den Modulen (Pfeil = importiert):

```
cli ─▶ config
    ─▶ data.build_mimicsql_db
    ─▶ data.convert ─▶ eval.hardness, eval.nl2sql360_tables, scripts.spider_get_tables,
    │                  nl2sql360.core
    ─▶ data.gold_check ─▶ eval.hardness, services.database_service
    ─▶ inference.runner ─▶ inference.prompts, inference.fewshot, inference.formats,
    │                      inference.values, inference.extract, services.ollama_service
    ─▶ eval.mimic_core ─▶ eval.executor ─▶ eval.hardness, services.database_service
    │                  ─▶ eval.nl2sql360_tables, inference.extract
    │                  ─▶ inference.runner (load_samples, load_records)
    ─▶ analysis.report ─▶ analysis.errors, eval.hardness (HARDNESS_ORDER),
                          inference.runner (load_samples)
```

---

## Konfiguration

### `config/config.yaml`

| Abschnitt | Bedeutung |
|---|---|
| `paths` | Quell-DB, MimicSQL2, TREQS, Ausgabeordner (`derived_dir`, `runs_dir`, `results_dir`, `docs_dir`) |
| `dataset` | Datensatzname und `db_id`, Fragefeld (`question_refine`), alle Splits, ausgewertete Splits (`eval_splits`) |
| `execution` | Timeout für SQL-Ausführung in Sekunden |
| `evaluation` | `ves_iterations`: Wiederholungen je richtiger Vorhersage für VES/R-VES (BIRD: 100, `0` = aus) |
| `report` | `contaminated_models`: Modelle mit bekannter Kontamination, nicht in „von wie vielen Modellen gelöst“ |
| `inference` | globale Ollama-Optionen: `temperature`, `seed`, `num_ctx`, `num_predict`, `keep_alive` |
| `variants` | Prompt-Varianten: `k` Beispiele, `strategy` (`random` / `similar`), optional `values: true` (Wert-Hinweise) |
| `selection` | `dev_variants` (Vergleich auf dev), `test_variant` (finaler Lauf auf test) |
| `models` | Liste von Modellen, klein nach groß; Reihenfolge bestimmt die Laufreihenfolge |

Felder pro Modell: `name` (Ordnername und Schlüssel in der CLI), `tag` (Ollama-Tag),
`format` (`chat`, `sqlcoder_v1`, `sqlcoder_v2`, `duckdb_nsql`), optional `think`, `num_ctx`,
`num_predict` und `extra_options` (weitere Ollama-Optionen, z.B. `num_batch` als Workaround
oder `stop` für die SQLCoder-Formate).

`config/format_check_sqlcoder.yaml` ist eine zweite, vollständige Config nur für den
Format-Check von SQLCoder (über `--config`); sie schreibt nach `runs/_format_check` und
`results/_format_check`.

Die Reihenfolge von Modellen und Varianten in der Config bestimmt auch die Reihenfolge in
allen Tabellen und Diagrammen.

### `src/config.py`

Lädt die YAML in Dataclasses. `Config` bündelt `Paths`, `DatasetConfig`,
`ExecutionConfig`, `EvaluationConfig`, `ReportConfig`, `InferenceConfig`, `VariantConfig`s, `SelectionConfig` und
`ModelConfig`s; das unveränderte YAML-Dict liegt zusätzlich in `Config.raw` und wird als
Snapshot ins Manifest jedes neuen Laufs geschrieben.

- `Paths` leitet alle Unterpfade ab: `db_path(db_id)`, `tables_file`, `split_file(dataset, split)`,
  `split_results_dir(split)`. Andere Module bauen keine Pfade selbst zusammen.
- `ModelConfig.options(inference)` erzeugt das `options`-Dict für Ollama. Werte am Modell
  überschreiben die globalen, `extra_options` kommen zuletzt dazu.
- `Config.model(name)` sucht ein Modell nach Namen (für `--models`).

### `src/cli.py`

`main()` parst die Argumente, lädt die Config, prüft `--split` (nur `dataset.eval_splits`,
also dev/test; train ist der Few-Shot-Pool), leitet loguru über `tqdm.write` um (damit
Log-Zeilen die Fortschrittsbalken nicht zerreißen) und verzweigt zum Subcommand. Die
Imports stehen in den Zweigen, damit jeder Schritt nur seine eigenen Abhängigkeiten lädt.

- `default_variants(config, split)`: auf dev alle `selection.dev_variants`, auf test genau
  `selection.test_variant`. Ist diese nicht gesetzt, bricht das Programm ab.
- `infer(...)`: erzeugt für jedes Paar aus Modell und Variante einen `InferenceRunner`,
  zählt über `runner.pending()` die offenen Antworten (Gesamtbalken) und arbeitet die
  Modelle der Reihe nach ab. Ein Modell ohne offene Antworten wird übersprungen. Nach den
  Varianten eines Modells wird es mit `unload()` aus dem VRAM entfernt.
- `existing_runs(...)`: alle gewählten Paare aus Modell und Variante, zu denen eine
  `raw.jsonl` existiert. Grundlage von `evaluate` und `reextract`.
- `evaluate(...)`: ruft `MimicCore.evaluate_run()` für jeden dieser Läufe auf.
- `reextract(...)`: ruft `reextract_run()` für jeden dieser Läufe auf und meldet die Zahl
  der geänderten Antworten.

---

## `build-db`: Datenbank nachbauen

### `src/data/build_mimicsql_db.py`

Baut die MIMICSQL-Datenbank (5 Tabellen, 100 Aufnahmen) nach dem TREQS-Verfahren. TREQS
erwartet MIMIC-III als CSV, vorhanden ist nur SQLite; daher der Umweg über CSV.

1. **Export** (`export_table_csv`): die neun Quelltabellen (`SOURCE_TABLES`) werden
   read-only gelesen und in `rowid`-Reihenfolge (= Reihenfolge des ursprünglichen
   CSV-Imports) als CSV geschrieben. `_format_value` schreibt ganzzahlige REAL-Werte ohne
   Nachkommastelle, wie in den Original-CSVs.
2. **TREQS-Tabellen** (`build_full_tables`): die unveränderten Funktionen aus
   `process_mimic_db/process_tables.py` werden importiert und erzeugen die vollen
   Tabellen. `_treqs_context` setzt dafür vorübergehend `cwd` und `sys.path`, weil TREQS
   relative Pfade nutzt (u.a. `id2name.csv`).
3. **Sampling** (`sample_admissions`): Nachbildung des Sampling-Teils von
   `process_mimic_db.py`, der dort beim Import ausgeführt würde. `DEMOGRAPHIC.sample(100,
   random_state=0)`, dann Filterung der übrigen Tabellen auf die gezogenen `HADM_ID`s.
4. **Indizes** (`add_indexes`) auf `HADM_ID` und `SUBJECT_ID`. Sie beschleunigen die
   Ausführung und ändern keine Ergebnisse.
5. Die DB entsteht als `.sqlite.tmp` und wird erst am Ende umbenannt; ein Abbruch
   hinterlässt also keine halbe DB. `build_manifest.json` hält Zeitpunkt, Seed und
   Zeilenzahlen fest. Zwischendateien unter `data/derived/_build/` werden gelöscht, außer
   mit `--keep-intermediate`.

Existiert die Ziel-DB, passiert ohne `--force` nichts.

---

## `prepare`: Datensatz aufbereiten

### `src/data/convert.py`

- `convert_split(config, split)` liest `new_{split}.json` (JSON Lines) und erzeugt pro
  Frage ein Sample (Format siehe [Datenformate](#konvertierte-splits)). `id` ist die
  Position im Split, `key` der Original-Schlüssel aus MimicSQL2. `spider_hardness` kommt
  aus `hardness.spider_hardness`, `parse_error`, wenn der Spider-Parser die Gold-SQL nicht
  lesen kann.
- `normalize_literals(sql)` erzeugt `sql_norm`: Doppelt gequotete Tokens werden von links
  gepaart; steht direkt davor ein `.`, ist es ein Bezeichner (`TABLE."COLUMN"`) und
  bleibt, sonst wird es ein Standard-SQL-Literal in einfachen Anführungszeichen.
  `sql_norm` wird nur für Few-Shot-Beispiele und die Fehleranalyse verwendet; EX und EM
  vergleichen gegen das Original-`sql`.
- `write_tables_json(config)` erzeugt `tables.json` im Spider-Format über
  `dump_db_json_schema` aus `src/scripts/spider_get_tables.py` (Original-Spider-Skript).
- Zum Schluss werden die `eval_splits` (dev, test) als NL2SQL360-Datasets
  `mimicsql_dev` / `mimicsql_test` importiert (`Core.import_dataset`). Als
  `sql_complexity_key` dient `spider_hardness`; die Schwierigkeit ist damit im
  NL2SQL360-Feld `complexity` filterbar. Ein Dataset mit vorhandenen Evaluationen wird
  nicht neu importiert (sonst müssten diese gelöscht werden).

### `src/eval/hardness.py`

Spider-Hilfen für Schwierigkeit und EM:

- `spider_canonical(schema, sql)`: schreibt Syntax, die der Spider-Parser nicht versteht,
  für Gold und Vorhersage identisch um: Anführungszeichen um Schema-Bezeichner entfernen,
  Spalten-Aliase im SELECT entfernen, `INNER/LEFT/CROSS JOIN` → `JOIN`.
- `parse_spider_sql(schema, sql)`: `(sql_dict, error)`; bei Fehler die leere SQL wie in
  NL2SQL360.
- `spider_hardness(schema, sql)`: easy / medium / hard / extra, `None` bei Parse-Fehler.
- `load_spider_schema(db_path)`: Spider-`Schema` aus der DB.
- `HARDNESS_ORDER`: Anzeigereihenfolge easy, medium, hard, extra, parse_error (Tabellen,
  Diagramme, `docs/gold_check.md`).

---

## `check-gold`: Gold-Queries prüfen

### `src/data/gold_check.py`

`check_gold` führt für jedes Sample aller Splits aus:

- `sql` und `sql_norm` auf der DB (`DBService`), Vergleich beider Ergebnisse mit
  `result_eq` → `sql_norm_equal`
- Parse mit sqlglot (`SQLParser` aus NL2SQL360) → `sqlglot_error`
- Parse mit dem Spider-Parser → `spider_parse_error`, `spider_hardness`

Ergebnis ist je Split `results/<split>/gold_check.csv` (Spalten siehe Code, keine
Ergebniszeilen) und der Bericht `docs/gold_check.md` (`write_report`: Übersicht je Split,
Fehlerarten, Spider-Parse-Fehler, Verteilung der Schwierigkeit in `HARDNESS_ORDER`, Splits
in Config-Reihenfolge).

Nicht ausführbare Gold-Queries schließt die Bewertung über `exec_acc = None` aus (siehe
`executor.py`).

---

## `infer`: Modelle befragen

### `src/inference/runner.py`

Ein `InferenceRunner` ist genau ein Lauf aus (Modell, Variante, Split) mit Ordner
`runs/<split>/<model>/<variant>/`. Beim Erzeugen baut er den `OllamaService`, den
Schematext, bei `k > 0` den `FewShotSelector` über den Train-Split und bei `values` den
`ValueMatcher`.

Ablauf von `run(limit, overall, label)`:

1. **Lock:** `run_lock` (`fcntl.flock` auf `.lock` im Laufordner). Läuft schon ein Prozess
   auf diesem Ordner, bricht der zweite ab (sonst doppelte Datensätze und verfälschte
   Latenzen).
2. **Offene Fragen:** `pending(limit)` lädt die Samples (ggf. die ersten `limit`) und
   entfernt alle, deren `key` schon in `raw.jsonl` steht. Darauf beruht das Fortsetzen.
3. **Konsistenzprüfung**, nur wenn es offene Fragen gibt: `_settings()` beschreibt alles,
   was bei einer Fortsetzung gleich bleiben muss: Modell-Config, Optionen, Variante,
   SHA-256 des Schemas und SHA-256 der gerenderten Prompt-Vorlage (`prompt_template`:
   der Prompt mit Platzhaltern statt Frage, Beispielen und Wert-Hinweisen). `_resumed_manifest()` prüft
   zuerst mit `_stored_prompts_match()`, dass die aktuelle Fassung jeden gespeicherten
   Prompt exakt reproduziert; das deckt auch Fragetexte und Few-Shot-Beispiele ab, die der
   Vorlagen-Hash nicht sieht. Dann vergleicht es die Einstellungen und den Modell-Digest.
   Bei einer Abweichung wirft es einen Fehler, bevor das Modell geladen wird. Manifesten
   ohne Prompt-Hash wird der aktuelle Hash ergänzt.
4. **Warm-up:** ein Aufruf mit dem längsten Prompt des Splits. Er lädt das Modell (die
   Ladezeit fällt so nicht in die gemessene Latenz). `_loaded_model` liest danach über
   `ollama ps`, ob das Modell vollständig im VRAM liegt (sonst Warnung) und mit welcher
   Kontextlänge es geladen ist. `check_loaded_context` bricht ab, wenn diese kleiner ist
   als `num_ctx`: Ollama kürzt ein `num_ctx` oberhalb der Kontextlänge des Modells ohne
   Fehlermeldung (`codellama:70b-instruct`: 2048 Tokens). `_check_context` prüft
   `prompt_tokens + num_predict ≤ num_ctx`. Danach wird das Manifest geschrieben (neu, mit
   Config-Snapshot) bzw. um einen `resumed`-Zeitstempel ergänzt.
5. **Hauptschleife:** pro offener Frage `render()` → `call()` → `_check_context()` →
   `extract_sql()` → Datensatz an `raw.jsonl` anhängen und sofort `flush()`. Bei einem
   Abbruch geht höchstens die laufende Frage verloren.
6. **Abschluss:** `write_predictions` schreibt `predictions.sql`, sobald alle Samples
   beantwortet sind (eine Zeile pro Frage in Datensatzreihenfolge, `NO_SQL_EXTRACTED` wenn
   nichts extrahiert wurde).

`call()` wählt anhand von `RenderedPrompt.mode` zwischen `chat` und `generate_raw`.

`reextract_run(config, split, path)` wendet `extract_sql` erneut auf den gespeicherten
`content` an (unter demselben Lock). Geänderte `pred_sql`/`extract_method` werden atomar in
`raw.jsonl` geschrieben (`.tmp` + `replace`), dazu `predictions.sql` und ein Eintrag
`reextracted` (`at`, `changed`) im Manifest. Prompts, Rohausgaben, Denk-Ausgaben und
Timings bleiben unverändert.

### `src/inference/prompts.py`

Modellunabhängiger Prompt-Inhalt:

- `INSTRUCTION` und `RULES` (SQLite-Dialekt, nur Schema-Spalten, nur SQL ausgeben)
- `COLUMN_DESCRIPTIONS`: Kurzbeschreibung jeder Spalte nach den MIMICSQL/TREQS-Definitionen
- `build_schema_text(db_path)`: `CREATE TABLE`-Statements aus `PRAGMA table_info` mit der
  Beschreibung als SQL-Kommentar und einem Hinweis auf `HADM_ID` als Join-Schlüssel. Es
  werden nur Metadaten gelesen.
- `PromptContent` (Instruktion, Regeln, Schema, Frage, Beispiele, Wert-Hinweise) mit
  `rules_text`, `examples_text` und `schema_text` (Schema plus Wert-Hinweise; ohne Hinweise
  exakt das Schema, sodass die übrigen Varianten unverändert rendern); `build_content()`
  setzt ihn zusammen.

### `src/inference/values.py`

`ValueMatcher(db_path)` lädt einmal die verschiedenen Werte der TEXT-Spalten der fünf
Tabellen, ohne Zeit- und Messwertspalten (`SKIP_COLUMNS`) und ohne reine Zahlen.
`hints(question)` liefert `TABLE.COLUMN = 'Wert'` für jeden Wert, der ohne
Groß-/Kleinschreibung an Wortgrenzen in der Frage steht, in DB-Schreibweise; ein Wert, der
in einem längeren Treffer derselben Spalte enthalten ist, entfällt. Die Werte bleiben im
Speicher und gehen nur in den Prompt; `raw.jsonl` speichert nur ihre Anzahl
(`n_value_hints`).

### `src/inference/fewshot.py`

`FewShotSelector.select(sample)` liefert `k` `Example`s aus dem Train-Split, jeweils mit
`sql_norm`:

- `random`: `random.Random(f"{seed}-{key}")`. Der Seed hängt an der Frage, nicht an der
  Reihenfolge; alle Modelle sehen zu einer Frage dieselben Beispiele, auch nach einem Resume.
- `similar`: TF-IDF (Uni- und Bigramme, sublineares TF) über die Train-Fragen, Kosinus-
  Ähnlichkeit (`linear_kernel`), stabile Sortierung. Das ähnlichste Beispiel steht zuletzt,
  direkt vor der Frage. Für Fragen, die identisch auch in train stehen, ist das die Lösung
  (siehe README, *Duplikate zwischen den Splits*).

### `src/inference/formats.py`

Rendert einen `PromptContent` in das Format des Modells (`RENDERERS`, ausgewählt über
`ModelConfig.format`):

| Format | Modus | Aufbau |
|---|---|---|
| `chat` | Chat-API | System: Instruktion + Regeln; User: Schema (+ Wert-Hinweise), Beispiele, `Question: … SQL:` |
| `sqlcoder_v1` | raw | `### Instructions:` (Anweisung, Regeln) / `### Input:` (Frage, Schema, `Examples`) / `### Response:`, endet mit einem geöffneten SQL-Codeblock. Layout von `defog/sqlcoder` (Ollama-Tag `sqlcoder:15b`) |
| `sqlcoder_v2` | raw | `### Task` / `### Instructions` / `### Database Schema` / `### Examples` / `### SQL`, endet mit einem geöffneten SQL-Codeblock. Layout von `defog/sqlcoder-7b` (Ollama-Tag `sqlcoder:7b`) und `sqlcoder2` |
| `duckdb_nsql` | raw | `### Instruction` / `### Input` / `Examples` / `### Question` / `### Response`, mit SQLite als Zieldialekt |

Bei den beiden SQLCoder-Formaten endet die Antwort am schließenden Zaun des Codeblocks.
Modelle mit diesen Formaten brauchen deshalb die drei Backticks als Stop-Sequenz
(`extra_options: {stop: […]}` in der Config, wie in Defogs `inference.py`). Ohne sie
schreibt das Modell hinter dem Zaun weiter, und die Extraktion könnte diesen Text als
Antwort nehmen. `tests/test_prompts.py` prüft das für alle Configs.

`RenderedPrompt.as_text()` liefert eine flache Textfassung, die in `raw.jsonl` gespeichert
und für den Vorlagen-Hash und die Resume-Prüfung verwendet wird.

### `src/services/ollama_service.py`

- `chat(messages)` und `generate_raw(prompt)` rufen den `ollama`-Client mit den Optionen,
  `think` (nur Chat) und `keep_alive` auf, messen die Wall-Clock-Zeit und geben eine
  `LLMResponse` zurück: Antworttext und Thinking getrennt, Token-Zahlen, Ollama-Zeiten
  (Nanosekunden → Sekunden), `done_reason`.
- `model_info()`: Tag, Digest, Größe, Details, Capabilities, Parameter und
  Kontextlänge für das Manifest.
- `unload()`: entlädt das Modell (`keep_alive=0`).
- `server_version()`: Ollama-Version über `/api/version`.

### `src/inference/extract.py`

`extract_sql(response)` gibt `(sql, method)` zurück:

1. `<think>…</think>` entfernen; bei nicht geschlossenem `<think>` alles ab dort verwerfen.
2. Quelle wählen, in dieser Reihenfolge: letzter Codeblock mit `SELECT`/`WITH`
   (`code_block`, auch ein nicht geschlossener Block), `[SQL]`-Tag (`sql_tag`), sonst
   Freitext ohne Fences (`plain`). SQL mit einem einzelnen schließenden Fence dahinter
   landet so im Freitext.
3. Bei Codeblock und `[SQL]`-Tag Kommentare (`--`, `/* */`) außerhalb von
   Anführungszeichen entfernen. Präfixe wie `SQL:` entfernen, ab dem ersten
   `SELECT`/`WITH` schneiden. Im Freitext zählen zuerst großgeschriebene Schlüsselwörter
   (Prosa schreibt „select“/„with“ klein), und er endet an der ersten Leerzeile.
4. `_first_statement`: Kommentare entfernen (sonst würde ein `--` nach dem Zusammenfassen
   der Zeilen den Rest der Query auskommentieren), bis zum ersten `;` außerhalb von
   Anführungszeichen schneiden, Whitespace außerhalb von Anführungszeichen zusammenfassen.

Ohne verwertbare SQL ist `sql = None` und `method` `empty` oder `no_sql`.

---

## `evaluate`: Bewerten

### `src/eval/mimic_core.py`

`MimicCore` erbt von `nl2sql360.core.Core` und ergänzt `evaluate_run(split, model,
variant, run_path, overwrite)`. Der NL2SQL360-Code selbst bleibt unverändert.

1. Samples und `raw.jsonl` laden. Weicht eine gespeicherte `pred_sql` von der aktuellen
   Extraktion ab, gibt es eine Warnung (dann zuerst `reextract`). Fehlen Antworten (z.B.
   Testlauf mit `--limit`), werden nur die vorhandenen bewertet.
2. Für jedes Sample `SampleEvaluator.score(pred_sql, sql)` und eine Ergebniszeile mit
   Score, Latenz, Tokens, Thinking-Länge und Schwierigkeit.
3. `add_efficiency`: Zeitverhältnis, VES und R-VES je Zeile (siehe
   `src/eval/efficiency.py`), dann `per_sample.csv` (immer neu geschrieben).
4. Nur bei vollständigen Läufen: Speichern in der NL2SQL360-Tabelle
   `DATASET_mimicsql_<split>_EVALUATION_<model>__<variant>` (`eval_name` ersetzt
   Sonderzeichen durch `_`, weil NL2SQL360 Tabellennamen ungequotet einsetzt). Eine
   vorhandene Tabelle wird nur mit `overwrite` ersetzt (`nl2sql360_tables.drop_evaluation`,
   ohne die interaktive Rückfrage von NL2SQL360). Befüllt werden `exec_acc`, `exact_acc`,
   `f1`, `ves` und `rves`; damit funktionieren `query_overall_performance` und die
   Leaderboards von NL2SQL360 auch für F1, VES und R-VES.

Der `SampleEvaluator` wird erst beim ersten Zugriff erzeugt (Property `evaluator`).

### `src/eval/executor.py`

`SampleEvaluator` bewertet eine einzelne Vorhersage mit den Spider-Bausteinen aus
NL2SQL360:

- **EX:** Gold (über `postprocess`, Ergebnis im Speicher gecacht) und Vorhersage
  ausführen, Vergleich mit `result_eq`; die Reihenfolge zählt nur bei `ORDER BY` in der
  Gold-SQL. Ergebnis `None`, wenn die Gold-SQL nicht ausführbar ist; `0`, wenn die
  Vorhersage fehlt oder fehlschlägt. Sind beide Ergebnisse leer, gilt das als gleich.
- **EM:** Spider-Parse beider SQLs, `rebuild_sql_val` / `rebuild_sql_col` mit der
  Fremdschlüssel-Map aus `tables.json`, dann `Evaluator.eval_exact_match`. Wie im
  Spider-Standard sind Werte und DISTINCT ausgeblendet (`DISABLE_VALUE`,
  `DISABLE_DISTINCT`). Ergebnis `None`, wenn die Gold-SQL nicht parsebar ist. Fehlt die
  Vorhersage, ist EM = 0 ohne Parse-Fehler. Ein Parse-Fehler der Vorhersage ergibt die
  leere SQL (also EM = 0) und wird in `em_parse_error_pred` vermerkt.
- **F1:** Soft F1 nach BIRD (`soft_f1` in `src/eval/metrics.py`) aus den Ergebniszeilen,
  die für EX ohnehin im Speicher liegen; keine zusätzliche Ausführung. `None` wie bei EX,
  `0`, wenn die Vorhersage fehlt oder fehlschlägt.

`SampleScore` enthält EX, EM, F1, Ausführungsstatus, Fehlermeldung, Zeilenzahlen, Laufzeit
und Parse-Fehler.

### `src/eval/metrics.py`

- `soft_f1(pred_rows, gold_rows)`: wie `calculate_f1_score` aus NL2SQL360 (BIRD
  mini-dev): Duplikate entfernen, Zeilen paarweise vergleichen, Zellen mit
  `calculate_row_match` (aus NL2SQL360 importiert), daraus Precision, Recall, F1.
  Einzige Änderung: BIRD paart die Zeilen in der Iterationsreihenfolge eines `set`, und
  die ändert sich bei Strings von Prozess zu Prozess (Hash-Randomisierung). Hier werden
  die Zeilen vorher stabil sortiert, der Wert ist also reproduzierbar. Beide Ergebnisse
  leer → 1, wie in BIRD.
- `rves_reward(time_ratio)`: Belohnungsstufen von `bird_rves` (Zeitverhältnis
  Gold/Vorhersage ≥ 2 → 1,25; ≥ 1 → 1; ≥ 0,5 → 0,75; ≥ 0,25 → 0,5; sonst 0,25; falsch → 0).
- `ves_score` = √Zeitverhältnis, `rves_score` = √Belohnung: die Werte pro Sample, die
  auch `VesEvaluator`/`RVesEvaluator` von NL2SQL360 speichern. Der Gesamtwert ist der
  Mittelwert × 100.

### `src/eval/efficiency.py`

`EfficiencyMeter.time_ratio(pred, gold)` misst wie `bird_ves.iterated_execute_sql`: Gold
und Vorhersage laufen abwechselnd `evaluation.ves_iterations`-mal (Standard 100), jeweils
auf einer neuen Verbindung (Verbindungsaufbau ungemessen, Cursor/Execute/Fetch/Close
gemessen). Das Zeitverhältnis ist der Mittelwert von Gold-Zeit / Vorhersage-Zeit nach dem
3σ-Ausreißerfilter `clean_abnormal` aus NL2SQL360. Gemessen werden nur Vorhersagen mit
EX = 1 (entspricht `reuse_ex=True`); falsche bekommen 0, Timeout oder Fehler ebenfalls 0.

Unterschiede zu NL2SQL360: Verbindung read-only mit echtem Timeout (Progress-Handler wie
`DBService`), sequenziell statt 8 parallele Prozesse (weniger Messrauschen), eine Messung
für VES und R-VES statt zwei.

`EfficiencyCache` speichert pro Lauf `efficiency.csv` (`key`, `measurement_hash` über
Iterationen, Vorhersage und Gold, `time_ratio`, `measured_at`). `evaluate` misst nur neu,
wenn sich der Hash geändert hat; Einträge von Fragen, die nicht mehr richtig sind,
fallen beim Speichern weg. `evaluation.ves_iterations: 0` schaltet die Messung ab
(VES/R-VES leer).

### `src/eval/nl2sql360_tables.py`

`drop_evaluation` und `drop_dataset` löschen NL2SQL360-Tabellen ohne die
`input()`-Rückfrage von `Core.delete_*_history` und entfernen die Tabellenklassen, damit
sie im selben Prozess neu angelegt werden können.

### `src/services/database_service.py`

`DBService` öffnet die DB read-only (`mode=ro`). `execute_sql(sql)` liefert ein
`ExecResult` (`ok`, `rows`, `error`, `runtime_s`, abgeleitet `row_count`, `timed_out`).

Der Timeout läuft über `set_progress_handler`: alle 10.000 SQLite-Instruktionen wird die
Deadline geprüft und die Abfrage in SQLite selbst abgebrochen (`interrupted` → Fehler
`timeout`). `rows` existiert nur im Speicher und wird nirgends gespeichert.

---

## `report`: Auswertung

### `src/analysis/report.py`

1. `load_per_sample`: alle `results/*/runs/*/*/per_sample.csv` zusammenführen.
2. `add_error_categories`: für jede falsche Antwort (`exec_acc == 0`) Kategorie und Flags
   über `categorize()` gegen `sql_norm`, plus Zuordnung zu einer Fehlergruppe
   (`ERROR_GROUPS`).
3. `add_train_duplicates`: Spalte `train_duplicate`, wenn Frage und Gold-SQL identisch auch
   in train stehen. `add_gold_tables`: Spalte `gold_tables` (Tabellen aus FROM/JOIN der
   Gold-SQL, mit `|` getrennt).
4. `summarize`: je (Split, Modell, Variante) `n`, `n_scored`, EX, EM, F1, VES, R-VES, `n_train_dup`,
   `EX_excl_train_dup` (EX ohne train-Duplikate; EX-Werte in %), Latenz (Mittel, Median),
   Prompt- und Output-Tokens, `no_sql`, `output_truncated`, EM-Parse-Fehler,
   `gold_excluded`.
5. Tabellen je Split unter `results/<split>/`: `all_samples.csv`, `summary.csv`,
   `summary.json`, `errors_by_model.csv`, `ex_by_spider_hardness.csv`,
   `errors_by_model_spider_hardness.csv`, `solved_by_models.csv` (je Frage: Anzahl und
   Namen der Modelle, die sie lösen; ohne `report.contaminated_models`).
6. `plot_split`: Diagramme als PNG und SVG unter `results/<split>/figures/`:

   | Datei | Inhalt |
   |---|---|
   | `ex_by_variant` | EX je Modell und Variante (nur wenn mehrere Varianten) |
   | `ex_em` | EX und EM je Modell |
   | `metrics_overview` | Heatmap Modell × EX, EM, F1, VES, R-VES |
   | `latency_vs_ex` | Median-Latenz gegen EX |
   | `ex_by_spider_hardness` | Heatmap EX nach Schwierigkeit |
   | `ex_by_table` | Heatmap EX nach Tabelle in der Gold-SQL (eine Frage zählt für jede ihrer Tabellen) |
   | `error_groups` | Fehlergruppen, Anteil aller Fragen |
   | `f1_wrong_answers` | Soft F1 der falschen Antworten (0 / unter 0,5 / ab 0,5 / 1), Anteil der falschen Antworten |
   | `rves_rewards` | R-VES-Belohnungsklassen der richtigen Antworten |
   | `value_hints_<variante>` | je Paar aus Variante mit `values` und gleicher Variante ohne: EX und Anteil der Fehler mit falscher Wert-Schreibweise, alle Modelle |
   | `ex_train_duplicates` | EX mit und ohne train-Duplikate |
   | `solved_by_models` | Fragen nach Anzahl der Modelle, die sie lösen, gestapelt nach Schwierigkeit; ohne `report.contaminated_models` |

   Außer `ex_by_variant` und `value_hints_*` zeigen auf dev alle Diagramme die beste
   Variante je Modell (nach EX), auf test die eine Test-Variante; der Titel nennt sie.

Reihenfolge: `_ranks`/`_sorted` ordnen Splits, Modelle und Varianten wie in der Config,
die Schwierigkeit nach `HARDNESS_ORDER` (easy, medium, hard, extra) und Fehlerkategorien
wie in `CATEGORIES`, in allen Tabellen und Diagrammen. Jede Fehlergruppe hat eine feste
Farbe (Position in `GROUP_ORDER`), damit dev und test vergleichbar aussehen.

### `src/analysis/errors.py`

`categorize(pred_sql, pred_error, gold_sql, schema_columns)` gibt
`(Hauptkategorie, Flags)` zurück:

- keine SQL → `no_sql`
- Ausführungsfehler → Kategorie aus der Fehlermeldung (`_exec_error_category`): `timeout`,
  `syntax_error`, `dialect_error` (`no such function`), `unknown_table`,
  `unknown_column`, `other_exec_error`
- falsches Ergebnis → `features()` zerlegt beide SQLs mit sqlglot (Tabellen, Spalten,
  Aggregationen, DISTINCT in Aggregationen, Joins und Join-Schlüssel, Literale,
  Vergleichsoperatoren). Jeder Unterschied wird ein Flag; `value_case`, wenn sich die
  Literale nur in der Groß-/Kleinschreibung unterscheiden. Die Hauptkategorie ist das
  erste Flag in der Reihenfolge von `CATEGORIES`, ohne Unterschied `other_wrong_result`.

Doppelt gequotete Tokens ohne Tabellenpräfix, die keine Schema-Spalte sind, zählen als
Literal (so interpretiert SQLite sie auch).

---

## Datenformate

### Konvertierte Splits

`data/derived/mimicsql_{split}.json`, Liste von Objekten:

| Feld | Inhalt |
|---|---|
| `id` | Position im Split (0-basiert) |
| `key` | Original-Schlüssel aus MimicSQL2; eindeutiger Bezug in allen Dateien |
| `split` | `train` / `dev` / `test` |
| `question` | Frage aus `dataset.question_key` |
| `sql` | Original-Gold-SQL |
| `sql_norm` | Gold-SQL mit einfachen Anführungszeichen für Literale |
| `db_id` | `mimicsql` |
| `spider_hardness` | easy / medium / hard / extra, `parse_error` wenn der Spider-Parser die Gold-SQL nicht liest |

### `runs/<split>/<model>/<variant>/manifest.json`

| Feld | Inhalt |
|---|---|
| `created` | Startzeitpunkt |
| `resumed` | Liste der Zeitpunkte, zu denen fortgesetzt wurde |
| `reextracted` | Liste von `{at, changed}`, ein Eintrag je `reextract` mit Änderungen |
| `split` | Split |
| `config` | Snapshot von `config/config.yaml` (Läufe ab dem 2026-11-02; ältere Manifeste haben das Feld nicht) |
| `settings` | `model`, `options`, `variant`, `schema_sha256`, `prompt_template_sha256` |
| `model_info` | Ausgabe von `OllamaService.model_info()` inkl. `digest` |
| `ollama_version` | Version des Ollama-Servers |
| `git` | `commit`, `dirty` |
| `gpu` | Angaben von `ollama ps` zum geladenen Modell: `size`, `size_vram`, `fully_on_gpu`, `context_length` (geladene Kontextlänge; ältere Manifeste haben das Feld nicht) |

### `runs/<split>/<model>/<variant>/raw.jsonl`

Eine Zeile pro beantworteter Frage:

| Feld | Inhalt |
|---|---|
| `key`, `id` | Bezug zum Sample |
| `prompt` | gerenderter Prompt (`RenderedPrompt.as_text()`) |
| `example_questions` | Fragen der Few-Shot-Beispiele |
| `n_value_hints` | Anzahl der Wert-Hinweise (nur Varianten mit `values`) |
| `content` | Rohantwort des Modells (ohne Thinking) |
| `thinking` | Denk-Ausgabe, getrennt; `null` bei Modellen ohne Reasoning |
| `prompt_tokens`, `output_tokens` | Token-Zahlen laut Ollama (`output_tokens` inkl. Reasoning) |
| `wall_s` | gemessene Zeit des Aufrufs |
| `total_duration_s`, `load_duration_s`, `prompt_eval_duration_s`, `eval_duration_s` | Zeiten laut Ollama |
| `done_reason` | z.B. `stop` oder `length` |
| `pred_sql`, `extract_method` | Ergebnis von `extract_sql`; `reextract` aktualisiert sie |
| `output_truncated` | `done_reason == "length"` |

### `results/<split>/runs/<model>/<variant>/per_sample.csv`

`id`, `key`, `model`, `variant`, `split`, `pred_sql`, `extract_method`, alle Felder von
`SampleScore` (`exec_acc`, `exact_acc`, `f1`, `pred_ok`, `pred_error`, `pred_row_count`,
`pred_runtime_s`, `gold_row_count`, `em_parse_error_gold`, `em_parse_error_pred`),
`latency_s`, `total_duration_s`, `prompt_tokens`, `output_tokens`, `thinking_chars`,
`output_truncated`, `spider_hardness`, `time_ratio`, `ves`, `rves`.

`report` ergänzt in `all_samples.csv` die Spalten `error_category`, `error_flags`
(mit `|` getrennt), `error_group`, `train_duplicate` und `gold_tables`.

### `results/<split>/runs/<model>/<variant>/efficiency.csv`

Cache der VES-Messung: `key`, `measurement_hash`, `time_ratio`, `measured_at`; nur
Fragen mit EX = 1. Löschen erzwingt eine neue Messung.

---

## Tests

| Datei | Prüft |
|---|---|
| `tests/test_extract.py` | SQL-Extraktion: Antwortformen, schließender Fence ohne öffnenden, Kommentare, Prosa vor und nach der SQL, letzter SQL-Block gewinnt, `WITH`, Whitespace, keine SQL |
| `tests/test_prompts.py` | Few-Shot-Auswahl (deterministisch je Key, bestes Beispiel zuletzt), gleicher Inhalt in allen Formaten, `normalize_literals`, Vorlagen-Hash, Wert-Hinweise (DB-Schreibweise, längster Treffer, Ausnahmen; Prompts ohne Hinweise unverändert) |
| `tests/test_errors.py` | Fehlerkategorisierung |
| `tests/test_executor.py` | EX/EM/F1 auf einer synthetischen Mini-DB, fehlende Vorhersage (EM = 0 ohne Parse-Fehler) |
| `tests/test_metrics.py` | Soft F1 (Teilpunkte, Reihenfolge, leere Ergebnisse), R-VES-Stufen, VES-Messung (read-only, Fehler → 0), Cache-Invalidierung |
| `tests/test_report.py` | Anzeigereihenfolge (Schwierigkeit easy → hard, Modelle nach Config) |

Ausführen mit `uv run pytest`.

## Nicht Teil des Ablaufs

- `src/scripts/spider_get_tables.py`: Original-Skript aus Spider; genutzt wird nur
  `dump_db_json_schema`, der `__main__`-Teil nicht.
