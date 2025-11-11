### Proyecto 3 - Mini-NMT: Encoder-Decoder y Beam Search

**Tema:** NMT con encoder-decoder, decodificación.

#### Objetivo
Construir un sistema de traducción automática neuronal **pequeño pero funcional** (EN -> ES) usando arquitectura **encoder-decoder con atención**.  
El foco principal es comparar **estrategias de decodificación**:  
- Greedy (beam=1)  
- Beam search con k=4 y k=8  
- Con y sin **length penalty** (α = 0.6 y α = 1.0)  

Se pide demostrar de manera  **cuantitativa y cualitativamente** cómo el **beam** mejora la fluidez y la precisión, pero también cómo puede generar traducciones demasiado cortas si no se penaliza la longitud.

#### Dataset
- **Recomendado**: Tatoeba EN-ES (disponible en Hugging Face: `tatoeba` o `Helsinki-NLP/tatoeba_mt`).  
  - ~150k pares de oraciones cortas y limpias. Ideal para fine-tuning rápido.  
- **Alternativa más desafiante**: Subset de WMT14/16 EN-ES (10k-50k pares).  
  - Descarga desde `statmt.org` o usa `datasets.load_dataset("wmt16", "en-es")`.  
- **Limpieza mínima obligatoria**:  
  - Eliminar pares vacíos o con >150 tokens.  
  - Normalización: `unicodedata.normalize("NFKC", text)` + lowercasing opcional.  
  - Tokenización con **SentencePiece** (modelo BPE de 8k-16k vocab).

#### Entregables
- **Notebook principal** con:  
  - Pipeline completo: carga -> limpieza -> tokenizador -> DataLoader -> modelo -> fine-tuning -> inferencia.  
  - Implementación **manual** de beam search (¡sin `torch.nn.functional.generate` ni `transformers.generate`!).  
  - Tabla comparativa automática de todas las configuraciones.  
- **Carpeta `results/`** con:  
  - `translations_comparison.csv` (10-15 oraciones de prueba con las 6 variantes: greedy, beam4, beam8, +penalidades).  
  - `metrics_table.png` o Markdown con BLEU/chrF por configuración.  
  - 3-5 ejemplos **cualitativos destacados** (oraciones donde **beam** gana claramente o donde **greedy** es mejor).  
- **Carpeta `figures/`** con:  
  - Gráfica de **BLEU vs beam size** (con y sin penalidades).  
  - Gráfica de **longitud media de traducción** vs configuración.  
  - Gráfica de **tiempo de inferencia por oración** (log scale).  
- **Video demo interactivo**.

#### Métricas
- **BLEU** (sacréBLEU para reproducibilidad).  
- **chrF** (más robusto con español).  
- **Tiempo medio por oración** (ms) en GPU/CPU.  
- Se espera:  
  - Greedy: BLEU ~22-26  
  - Beam 4 + penalty 0.6: BLEU ~29-32  
  - Beam 8 + penalty 1.0: BLEU ~30-33 (punto típico).

#### Pasos (desglosados con detalle)
1. **Limpieza/normalización básica**  
   - Usa `datasets` de Hugging Face.  
   - Filtra oraciones >120 tokens en origen o destino.  
   - Entrena un tokenizador **SentencePiece BPE** con vocab 16k (un solo modelo para EN+ES o dos por separado).  
   - Añade tokens especiales: `<s>`, `</s>`, `<pad>`, `<unk>`.  
   - Durante batching: padding + `src_mask` y `tgt_mask` (causal para decoder).

2. **Fine-tune (2-3 épocas)**  
   - Modelo: **Transformer pequeño** (6 capas encoder, 6 decoder, d_model=256, heads=8, d_ff=1024).  
   - Puedes partir de pesos aleatorios o de `facebook/m2m100_418M` congelando embeddings (más rápido).  
   - Loss: `LabelSmoothedCrossEntropy` (ε=0.1).  
   - Optimizer: Adam con lr=5e-4, warmup 400 steps.  
   - Batch size efectivo ~4096 tokens (acumula gradientes si es necesario).  
   - Guardado: solo el mejor por valid BLEU.

3. **Decodificar y comparar búsquedas**  
   - Implementa **beam search desde cero**:  
     - Mantén `beam_size` hipótesis activas.  
     - Normaliza scores por longitud **solo si** length_penalty > 0:  
       `score = log_prob / (len**α)`  
     - Early stopping con EOS.  
   - Prueba 6 configuraciones:  
     - greedy (k=1, α=0)  
     - beam4 α=0  
     - beam4 α=0.6  
     - beam8 α=0  
     - beam8 α=0.6  
     - beam8 α=1.0  
   - Evalúa con `sacremoses` + `sacrebleu`.

#### Video
**Guion recomendado**:  
1. **Introducción**: "Hoy construimos un traductor EN->ES desde cero y vemos por qué el beam search de ChatGPT no es magia... es solo búsqueda inteligente."  
2. **Demostración en vivo**:  
   - Escribe 5 oraciones en inglés (una fácil, una ambigua, una larga, una con modismo, una técnica).  
   - Muestra en tiempo real cómo cambia la traducción al activar/desactivar beam y la penalidad.
3. **Resultados**: tabla BLEU, gráfica de longitud, tiempo.  
4. **Conclusiones**:  
   - "Beam > greedy siempre, pero sin length penalty traduce demasiado corto."  
   - "α=0.6 es el sweet spot para EN-ES."  
   - "Beam 8 es solo marginalmente mejor que 4, pero 2× más lento."  
   - Cierre: "Ahora sabes exactamente cómo funciona el botón 'traducir' de los LLMs."

### Ejecutar con Docker
> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):  
   ```bash
   make build
   ```
   -> Crea la imagen `p3_mininmt_beamsearch`.

2. **Levantar entorno con Jupyter** (mapea el proyecto en `/workspace`):  
   ```bash
   make jupyter
   # Abrir en el navegador: http://localhost:8888
   ```

3. **Shell dentro del contenedor** (opcional, para ejecutar scripts):  
   ```bash
   make sh
   ```

4. **Detener** todo:  
   ```bash
   make stop
   ```

#### Notas importantes
- El volumen `-v $PWD:/workspace` asegura que **todo queda guardado localmente**.  
- `requirements.txt` debe incluir:  
  `torch`, `torchtext`, `datasets`, `sentencepiece`, `sacrebleu`, `sacremoses`, `tqdm`, `matplotlib`, `seaborn`, `jupyterlab`, `pandas`.  
- Usa GPU: el fine-tuning de 2 épocas con 150k pares dura ~15-20 min en una RTX 3060.

<details><summary>Ejemplo (opcional) con GPU CUDA - más explicado</summary>

**Por qué GPU es casi obligatorio aquí**:  
- Fine-tuning sin GPU: >2 horas.  
- Con GPU: 15-20 min total.  
- Inferencia con beam 8: 50 ms/oración (vs 800 ms en CPU).

**Dockerfile.gpu** (copia tal cual):
```Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git curl tini && rm -rf /var/lib/apt/lists/*
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /workspace
COPY --chown=appuser:appuser requirements.txt .
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''"]
```

**Comandos para GPU**:
```bash
# 1. Construir (solo primera vez)
docker build -t p3_mininmt_beamsearch:gpu -f Dockerfile.gpu .

# 2. Ejecutar con GPU
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace --name nmt_gpu p3_mininmt_beamsearch:gpu

# 3. Abrir http://localhost:8888
# 4. Detener
docker stop nmt_gpu
```

Verifica GPU en notebook:
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))
```
</details>
