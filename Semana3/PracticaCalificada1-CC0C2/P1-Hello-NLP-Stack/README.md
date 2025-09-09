### Proyecto 1-Hello NLP Stack (Clasificador baseline reproducible)

**Objetivo**: baseline TF-IDF + Regresión Logística con **F1 macro** y split reproducible.

#### Datos
```bash
make data
make checks
```
#### Pipeline
```bash
make preprocess
make train
make eval
make report
```
**Implementa** limpieza en `preprocess.py`, baseline en `train.py`, métricas (acc/F1 y `cm.png`) en `eval.py`.

**Análisis**: Curva tamaño vs F1 (≥3 puntos) y mejora ≥+20% vs *majority class*.


#### Controles (obligatorios)

- Commits en **español**: `git config commit.template .gitmessage`
- **Comentarios** del código en **español** (se auditan ≥2 archivos)
- **Evidencias**: `make checks && make verify-data` -> `evidencias/data_sha256*.txt`
- **Anti "última hora"**: ≥3 PRs (12/15/18) y <60% de cambios 19-20 septiembre.

#### Uso rápido

```bash
pip install -r requirements.txt
make data
make preprocess
make train
make eval
make report
make checks
```

 #### Entrega

- `make data` ejecutado, CSV presentes en `data/raw/`.
- `make checks && make verify-data` y subida de `evidencias/data_sha256.txt`.
- Pipeline completo corre en limpio: `preprocess -> train -> eval -> report`.
- Comentarios en español, commits en español.
- Artefactos: `out/metrics.json`, figuras, `docs/reporte.md`, **slides**, **preguntas y respuestas** (no oral), **video ≥5 min**.

