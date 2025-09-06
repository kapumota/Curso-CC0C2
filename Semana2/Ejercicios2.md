### **Ejercicios 2 CC0C2**

**Cuadernos base:** `Librerias_NLP.ipynb` y `Cargador_datos_NLP.ipynb`  
**Objetivo general:** practicar, comparar y reforzar la ingeniería de datos para NLP (tokenización, vocabularios, colación y *padding*) 
y el uso de modelos HF (generación/prompting), con una progresión de ejercicios cortos y otros más retadores.

#### Entregables y formato sugerido
- Carpeta del alumno/equipo: `Actividad-NLP/`
  - `Librerias_NLP/` -> evidencias E1-E7 (pantallazos o celdas ejecutadas).
  - `Cargador_datos_NLP/` -> evidencias E8-E15 y E16+ (más retadores).
  - `README.md` breve (5-10 líneas): qué funcionó, qué no, decisiones y hallazgos.
- Evidencias numeradas: `evidencias/01----*.txt|png|ipynb` (estilo consistente).
- Roles (si trabajan en parejas): *driver* (teclea), *navigator* (guía), *reporter* (anota).
- **Regla de oro:** no mezcles tokenizadores y modelos de checkpoints distintos.

#### Parte A - Mini-labs con `Librerias_NLP.ipynb`

> Meta: experimentar con carga de modelos/tokenizadores HF, parámetros de generación y bucles de chat, observar determinismo vs creatividad y controlar longitud.

#### Ejercicio 1 - Verificación de entorno
**Objetivo:** asegurar intérprete y versiones.  
**Pasos:** ejecuta `print(sys.executable)`, `transformers.__version__`, `sentencepiece.__version__`.  
**Entrega:** ruta de Python y versiones (una línea).  
**Mini-check:** ¿coinciden versiones dentro del equipo?


#### Ejercicio 2 - Carga correcta de par modelo/tokenizador
**Objetivo:** cargar **BlenderBot 400M distill** o **FLAN-T5 base** y validar el par.  
**Pasos:** `AutoTokenizer.from_pretrained(model_name)` y `AutoModelForSeq2SeqLM.from_pretrained(model_name)`. Genera una respuesta a "Hola desde Lima".  
**Entrega:** `type(tokenizer)`, `type(model)` y primer output.  
**Mini-check:** si cambiaste el *checkpoint*, ¿cambiaste el tokenizador?


#### Ejercicio 3 - Decodificación: determinista vs creativa
**Objetivo:** comparar *beam search* vs *sampling*.  
**Pasos:** mismo *prompt* con (a) `num_beams=5, do_sample=False` y (b) `do_sample=True, top_p=0.9, temperature=0.8`. Repite (b) dos veces.  
**Entrega:** 3 outputs + comentario (2-3 líneas).  
**Mini-check:** (a) estable, (b) varía entre corridas.


#### Ejercicio 4 - Control de longitud y costo
**Objetivo:** observar efecto de `max_new_tokens`.  
**Pasos:** genera con 20 y 120, mide el tiempo "aprox." con `time.time()`.  
**Entrega:** longitudes y tiempos, reflexión (1-2 líneas).


#### Ejercicio 5 - Bucle de chat robusto
**Objetivo:** completar `chat_with_bot()` con: salida (`exit/quit/bye`), manejo de excepciones, límite de *prompt*.  
**Entrega:** 3-4 turnos + evidencia de salida controlada.  
**Mini-check:** ¿qué haces con entrada vacía?


#### Ejercicio 6 - *Prompting* de tareas con FLAN-T5
**Objetivo:** probar instrucciones tipo `summarize:` y `translate Spanish to English:`.  
**Entrega:** outputs y comentario breve de calidad.  
**Mini-check:** ajusta `max_new_tokens` si corta o divaga.


#### Ejercicio 7 - Español: comparación rápida
**Objetivo:** comparar ES en BlenderBot vs FLAN-T5.  
**Entrega:** tabla mini (modelo, *score* 1-5, nota corta).  
**Mini-check:** ¿cuál rinde mejor y por qué?


#### Parte B - Mini-labs con `Cargador_datos_NLP.ipynb`

> Meta: dominar tokenización, vocabularios, `collate_fn`, *padding*, BOS/EOS y batching (Multi30k). Comprender `batch_first` y la eficiencia de ordenar por longitud.

#### Ejercicio 8 - Dataset mínimo y DataLoader
**Objetivo:** crear `CustomDataset` (devuelve cadenas) y `DataLoader`.  
**Pasos:** 6-8 oraciones, `__len__`, `__getitem__`, itera 2 lotes.  
**Entrega:** pantallazo de lotes.  
**Mini-check:** *shuffle* cambia el orden.


#### Ejercicio 9 - Tokenizador + vocabulario
**Objetivo:** convertir a tensores de índices.  
**Pasos:** `get_tokenizer("basic_english")`, `build_vocab_from_iterator(map(tokenizer, sentences), specials=[...])`,  dataset devuelve `torch.tensor(ids)`.  
**Entrega:** 2 ejemplos (tensor + lista de tokens).  
**Hint:** `vocab.get_itos()` para reconstruir.


#### Ejercicio 10 - `collate_fn` con `pad_sequence` (batch_first=True)
**Objetivo:** homogeneizar longitudes del lote.  
**Pasos:** implementa `collate_fn`. Muestra `shape` y lote *padded*.  
**Entrega:** `shape` y lote, identifica `PAD_IDX`.  
**Mini-check:** usa `padding_value=PAD_IDX`.


#### Ejercicio 11 - `batch_first=False`
**Objetivo:** ver `(T, B)`.  
**Pasos:** `collate_fn_bfFALSE`, muestra `shape` y lote.  
**Entrega:** comparación `(B, T)` vs `(T, B)` y preferencia (1 línea).


#### Ejercicio 12 - Multi30k: primer vistazo
**Objetivo:** cargar *train iterator* y revisar (DE, EN).  
**Pasos:** `Multi30k(split='train', language_pair=('de','en'))`,  `next(iter(...))`, imprime.  
**Entrega:** ejemplo visible.  
**Hint:** si caen URLs, redefine `multi30k.URL[...]` al espejo indicado.


#### Ejercicio 13 - spaCy tokenizers por idioma
**Objetivo:** tokenizar con mejor calidad.  
**Pasos:** descarga `de_core_news_sm` y `en_core_web_sm`, `get_tokenizer('spacy', language=...)` para DE/EN.  
**Entrega:** listas de tokens por idioma.  
**Mini-check:** diferencias vs `basic_english`.

#### Ejercicio 14 - Vocabs por idioma + BOS/EOS
**Objetivo:** construir `vocab_transform[de]` y `[en]` + añadir BOS/EOS.  
**Pasos:** `build_vocab_from_iterator(yield_tokens(...), specials=['<unk>','<pad>','<bos>','<eos>'], special_first=True)` y `set_default_index(UNK_IDX)`. Implementa `tensor_transform_s` (flip src) y `tensor_transform_t` (tgt).  
**Entrega:** 1 ejemplo con índices (incluye BOS/EOS).  
**Mini-check:** comenta si mantendrías el *flip* al usar Transformers.


#### Ejercicio 15 - `collate_fn` (pares) + DataLoader train/valid
**Objetivo:** empaquetar (src, tgt) con *padding* y verificar *shapes*.  
**Pasos:** `collate_fn_translation` con `pad_sequence(..., padding_value=PAD_IDX, batch_first=True)`, `BATCH_SIZE=4`, orden por longitud, `drop_last=True`.  
**Entrega:** `src.shape`, `tgt.shape` y 1 lote truncado (vista parcial).  
**Mini-check:** ¿disminuyó el *padding waste* al ordenar por longitud?

#### Parte C - Ejercicios **más retadores** 


#### Ejercicio 16 - *Bucketing Sampler* por longitud

**Objetivo:** reducir *padding waste* agrupando ejemplos por rangos de longitud antes de formar lotes.  
**Tareas:**
1) Implementa un *sampler* (o pre-bucketización previa) que cree grupos por longitud (bins de 10 tokens).  
2) Compara `ratio_pad = (#PAD)/(#tokens totales)` con y sin *bucketing* en una época.  
**Entrega:** tabla comparativa (sin/ con bucketing) y % de mejora.

#### Ejercicio 17 - Métrica de *padding waste* + logging por lote
**Objetivo:** instrumentar medición continua.  
**Tareas:**
1) En cada batch, calcula `pad_count` y `token_count`.  
2) Acumula por época y registra p50/p90/p99 del *waste*.  
**Entrega:** resumen y breve análisis (2-3 líneas).


#### Ejercicio 18 - Máscaras de atención y máscara causal del *decoder*
**Objetivo:** derivar `attention_mask` (1=token, 0=PAD) y máscara causal triangular para el *decoder*.  
**Tareas:**
1) A partir de `src_batch`/`tgt_batch` *padded*, genera `src_attn_mask` y `tgt_attn_mask`.  
2) Crea `tgt_causal_mask` (triangular) para impedir mirar tokens futuros.  
**Entrega:** *shapes* y verificación de diagonales/superioridad nula.


#### Ejercicio 19 - *Loss* ignorando PAD + *teacher forcing* (bucle mínimo)
**Objetivo:** preparar un entrenamiento minimalista.  
**Tareas:**
1) Implementa una función de *loss* que ignore `PAD_IDX` (usar `ignore_index`).  
2) Arma un bucle simulado de entrenamiento con *teacher forcing* (no requiere modelo complejo,  puede ser *dummy* que solo pruebe la forma).  
**Entrega:** log de 1-2 iteraciones con *loss* bajando artificialmente (si aplicas un truco simple).


#### Ejercicio 20 - Sustituir TorchText por *tokenizador HF* (compatibilidad de checkpoint)
**Objetivo:** usar el *tokenizer* del modelo (p.ej., T5/mT5) para garantizar compatibilidad.  
**Tareas:**
1) Reemplaza `token_transform`/`vocab_transform` por `AutoTokenizer` del checkpoint.  
2) Asegura `padding=True`, `truncation=True`, `return_tensors='pt'` y crea una `collate_fn` análoga.  
**Entrega:** lote listo para el modelo HF y comparación de *shapes* con la versión TorchText.


#### Ejercicio 21 - Portar a `datasets` (HF) + `map` + `DataCollatorWithPadding`
**Objetivo:** comparar pipelines.  
**Tareas:**
1) Carga un *split* compatible (o usa un dataset de texto de HF).  
2) Aplica `map` con el tokenizador y usa `DataCollatorWithPadding`.  
3) Compara tiempos/ergonomía vs TorchText.  
**Entrega:** observaciones (3-5 líneas) y *snippet* clave.


#### Ejercicio 22 - *Tests* mínimos (pytest) para el *pipeline* de datos
**Objetivo:** validar contratos.  
**Tareas:** crea *tests* que verifiquen:  
- (a) presencia de BOS/EOS,  
- (b) `PAD_IDX` solo en cola del *padding*,  
- (c) *shapes* esperados y `dtype=int64`,  
- (d) reversa aplicada solo a `src`.  
**Entrega:** salida de `pytest -q` y 1-2 tests mostrados.


#### Ejercicio 23 - Script CLI de muestreo de lotes
**Objetivo:** exponer el *pipeline* como herramienta.  
**Tareas:** con `argparse`, crea `sample_loader.py` que imprime `N` ejemplos *tokenizados* y un lote.  
**Entrega:** uso (`python sample_loader.py --n 3 --batch-size 4`) y salida.


#### Ejercicio 24 - *Feature flag* de tokenización (spaCy vs básica vs HF)
**Objetivo:** alternar modos sin tocar lógica central.  
**Tareas:** agrega una bandera de entorno o argumento CLI para escoger tokenizador.  
**Entrega:** evidencias de los 3 modos y diferencias de *tokens*.


#### Ejercicio 25 - Truncado seguro + histograma de longitudes
**Objetivo:** evitar OOM y entender distribución.  
**Tareas:** define `max_len` y trunca. Genera un histograma (binning) de longitudes *antes/después*.  
**Entrega:** histograma textual y reflexión (2-3 líneas).


#### Ejercicio 26 - `collate_fn` (diccionarios tipados)
**Objetivo:** preparar datos estilo producción.  
**Tareas:** devuelve `{"src": ..., "tgt": ..., "src_mask": ..., "tgt_mask": ...}` y anota *type hints*.  
**Entrega:** `print(batch.keys())` y *shapes*.


#### Ejercicio 27 - Reproducibilidad total
**Objetivo:** fijar semillas y versiones.  
**Tareas:** `random.seed`, `np.random.seed`, `torch.manual_seed`, control de *workers* deterministas y listado de versiones (`pip freeze | grep -E 'torch|torchtext|spacy|transformers'`).  
**Entrega:** bloque `SEEDS` y lista de versiones (4-6 líneas).

#### Ejercicio 28 - Métrica de cobertura de vocabulario (OOV rate)
**Objetivo:** medir cuántos tokens quedan fuera de vocab.  
**Tareas:** calcula `%OOV` por *split*. Analiza impacto de `min_freq`.  
**Entrega:** tabla (split, min_freq, %OOV) y comentario (2-3 líneas).

#### Pistas generales (no soluciones)
- Mantén alineado *tokenizer <-> modelo*.  
- Revisa `padding_value` vs `PAD_IDX`.  
- Inspecciona `itos` para depurar. Imprime 1-2 secuencias "reconstruidas".  
- Ordenar por longitud reduce *padding waste*. *bucketing* lo reduce aún más.  
- En Transformers, no necesitas *flip* de la fuente.  
- Si no hay internet, usa `basic_english` o un tokenizador HF previamente *cacheado*.

