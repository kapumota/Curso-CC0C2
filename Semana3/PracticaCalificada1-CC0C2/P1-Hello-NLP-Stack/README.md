# P1 — Hello NLP Stack (Clasificador baseline reproducible)

**Objetivo**: baseline TF-IDF + Regresión Logística con **F1 macro** y split reproducible.

### Datos
```bash
make data
make checks
```
### Pipeline
```bash
make preprocess
make train
make eval
make report
```
**Implementa** limpieza en `preprocess.py`, baseline en `train.py`, métricas (acc/F1 y `cm.png`) en `eval.py`.

**Análisis**: curva tamaño vs F1 (≥3 puntos) y mejora ≥+20% vs *majority class*.


## Calendario y entregables

- **Zona horaria:** America/Lima
- **Hitos:** Hito 1 → 12-sep 23:59 · Hito 2 → 15-sep 23:59 · *Code freeze* → 18-sep 18:00 · **Entrega final** → 20-sep 23:59
- **Entregables:**
  - **Trabajo (3 pts)** — código + README reproducible + figuras/tablas.
  - **Video sprint (4 pts, ≥5 min)** — objetivo → backlog/hitos → demo E2E → métricas → lecciones.
  - **Exposición + preguntas (13 pts)** — slides con notas + cuestionario **escrito** (no oral).

## Controles (obligatorios)

- Commits en **español**: `git config commit.template .gitmessage`
- **Comentarios** del código en **inglés** (se auditan ≥2 archivos)
- **Símbolo creado por IA** en `docs/` + herramienta y prompt
- **Evidencias**: `make checks && make verify-data` → `evidencias/data_sha256*.txt`
- **Anti “última hora”**: ≥3 PRs (12/15/18) y <60% de cambios 19–20 sep

## Uso rápido

```bash
pip install -r requirements.txt
make data
make preprocess
make train
make eval
make report
make checks
```

## FAQ

- **HF lento:** usa `make data N=2000` (o `N_WIKI=5000`)
- **Windows:** WSL (Ubuntu); ejecuta dentro de Linux
- **Makefile:** recetas usan **TAB**
- **UD (P4):** coloca `.conllu` en `data/external/UD_Spanish-GSD/`
- **Versiones:** `pip freeze > evidencias/pip-freeze.txt`


---

## ✅ Checklist de entrega

- [ ] `make data` ejecutado; CSV presentes en `data/raw/`.
- [ ] `make checks && make verify-data` y subida de `evidencias/data_sha256.txt`.
- [ ] Pipeline completo corre en limpio: `preprocess → train → eval → report`.
- [ ] Comentarios en inglés; commits en español (plantilla activada).
- [ ] 3 PRs (12/15/18) y <60% de cambios 19–20 sep.
- [ ] Artefactos: `out/metrics.json`, figuras, `docs/reporte.md`, **slides**, **símbolo IA**, **preguntas y respuestas** (no oral), **video ≥5 min**.
- [ ] Específico del proyecto cumplido (P1…P7) según README.
