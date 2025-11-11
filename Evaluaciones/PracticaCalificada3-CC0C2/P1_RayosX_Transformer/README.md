### Proyecto 1 - Rayos X del Transformer
**Tema:** Fundamentos (MHSA, FFN, residual, Norm, máscaras).

#### Objetivo
Implementar **desde cero** un bloque Transformer mínimo (encoder o decoder, según prefieras) que sea 100 % funcional para clasificación de texto. El foco principal es **entender y demostrar** cómo funcionan internamente dos tipos de máscaras:
- **Máscara de padding**: para ignorar los tokens `<pad>` en secuencias de longitud variable.
- **Máscara causal** (look-ahead): para evitar que el modelo "vea el futuro" durante el entrenamiento autoregresivo (útil si decides usar arquitectura tipo decoder).

Además, deberás **visualizar los mapas de atención** (attention weights) de la capa MHSA en al menos tres ejemplos distintos del conjunto de validación, mostrando claramente qué palabras atiende el modelo y por qué.

#### Dataset
- **Opción recomendada**: IMDb (clasificación binaria de sentimiento, ~50 k reseñas). Es más sencillo para empezar y permite entrenar rápido.
- **Opción alternativa**: AG News (4 clases: World, Sports, Business, Sci/Tech). Útil si quieres practicar con más clases y ver cómo se comporta la atención en temas distintos.

Ambos datasets están disponibles directamente en `torchtext` o `datasets` (Hugging Face).

#### Entregables
- **Notebook principal** (.ipynb) con:
  - Todo el código del bloque Transformer implementado manualmente (sin usar `nn.Transformer` ni `TransformerEncoderLayer`).
  - Pipeline completo: carga de datos -> tokenización -> DataLoader con `attention_mask` -> entrenamiento -> evaluación.
  - Visualizaciones claras y bien comentadas.
- **Carpeta `figures/`** con:
  - Al menos **3 mapas de atención** (heatmap) de diferentes ejemplos (uno positivo, uno negativo, uno ambiguo o de clase distinta si usas AG News).
  - Gráficos de **pérdida de train/val** y **accuracy** por época.
- **Carpeta `metrics/`** con un archivo `final_metrics.json` que contenga:
  ```json
  {
    "test_accuracy": 0.XXX,
    "test_f1": 0.XXX,
    "val_loss": X.XXX,
    "train_loss": X.XXX,
    "best_epoch": X
  }
  ```
- **Video** (grabación de pantalla + voz o texto):
  - Introducción rápida al proyecto.
  - Demo en vivo de los mapas de atención (explicar qué está mirando el modelo).
  - Mostrar curvas de entrenamiento y métricas finales.
  - Cierre con 3-4 lecciones aprendidas (ejemplo: "la máscara causal es clave para tareas generativas", "el padding mask evita que el modelo preste atención a tokens vacíos", etc.).

#### Métricas
- Clasificación: **Accuracy** y **F1-score** (macro para multiclasse).
- Pérdida: **CrossEntropyLoss** en train y validation.
- Se espera alcanzar al menos **85 % accuracy** en IMDb con 3-5 épocas (es totalmente factible con un bloque bien implementado).

#### Pasos (desglosados con más detalle)
1. **Tokenización + batching con `attention_mask`**  
   - Usa `torchtext` o `transformers` (BertTokenizer es válido si solo quieres el tokenizer).  
   - Asegúrate de devolver `input_ids` y `attention_mask` (1 para tokens reales, 0 para padding).  
   - En el `DataLoader` usa `collate_fn` personalizado para crear batches con padding dinámico y generar la máscara correctamente.

2. **MHSA (scaled dot-product) + máscara causal opcional**  
   - Implementa la atención multi-cabecera **manualmente**:  
     - Proyección lineal de Q, K, V.  
     - `attn_scores = (Q @ K.transpose(-2,-1)) / sqrt(d_k)`  
     - Aplicar **padding mask** (sumar `-inf` donde `attention_mask == 0`).  
     - Si usas decoder: aplicar también **máscara causal** (triangular superior con `-inf`).  
   - Softmax -> pesos de atención -> salida = pesos @ V.

3. **FFN (GeLU/SiLU), residual, (RMS)Norm, Dropout**  
   - Feed-Forward: dos lineales con expansión 4x (ej. 768 -> 3072 -> 768).  
   - Activación: prueba **GeLU** (más común) o **SiLU** (Swish).  
   - Residual connection alrededor de MHSA y alrededor de FFN.  
   - Normalización: puedes usar `LayerNorm` o `RMSNorm` (más moderna y sin bias).  
   - Dropout después de cada sub-capa (p=0.1 es un buen punto de partida).

4. **Entrenar 3-5 épocas y graficar**  
   - Optimizer: AdamW con weight decay 0.01.  
   - Learning rate: 5e-4 o 1e-3 (puedes usar scheduler lineal con warmup si quieres).  
   - Early stopping opcional si la validación no mejora en 2 épocas.  
   - Guarda el modelo con mejor `val_loss`.  
   - Grafica **train loss**, **val loss**, **train acc**, **val acc** en la misma figura (dos ejes y).

#### Video
**Guion recomendado**:  
1. **Sprint intro**: "Este proyecto es un 'rayos X' del Transformer: vamos a construir un bloque desde cero y ver exactamente qué pasa dentro de la atención."  
2. **Demostración de atención**:  
   - Muestra 3 ejemplos reales.  
   - Explica: "Aquí el modelo pone mucho peso en la palabra 'horrible' cuando predice negativo."  
   - Compara con y sin máscara causal (si la implementaste).  
3. **Métricas**: muestra las curvas y el JSON final.  
4. **Cierre**: 3 lecciones clave + "¡Listo para el siguiente proyecto!".

### Ejecutar con Docker
> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):  
   ```bash
   make build
   ```
   -> Crea la imagen `p1_rayosx_transformer` con todas las dependencias.

2. **Levantar entorno con Jupyter** (mapea el proyecto en `/workspace`):  
   ```bash
   make jupyter
   # Abrir en el navegador: http://localhost:8888
   ```
   -> Todo lo que guardes en tu carpeta local queda persistente.

3. **Shell dentro del contenedor** (opcional, para ejecutar scripts o debug):  
   ```bash
   make sh
   ```

4. **Detener** todo:  
   ```bash
   make stop
   ```

#### Notas importantes
- Gracias al volumen `-v $PWD:/workspace`, **nunca pierdes tu trabajo** aunque borres el contenedor.
- El `requirements.txt` debe incluir: `torch`, `torchtext`, `matplotlib`, `seaborn`, `numpy`, `tqdm`, `datasets` (opcional), `jupyterlab`.
- Si tu máquina tiene GPU y quieres usarla, sigue el ejemplo CUDA que ya está expandido más abajo.

<details><summary>Ejemplo (opcional) con GPU CUDA - más explicado</summary>

**Por qué usar una imagen CUDA**:  
- Entrenamiento 10-20× más rápido.  
- Con CPU puedes tardar 20-30 min por época; con GPU < 2 min.

**Dockerfile.gpu** (copia tal cual, solo cambia la versión de CUDA si necesitas):
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
# 1. Construir (solo la primera vez)
docker build -t p1_rayosx_transformer:gpu -f Dockerfile.gpu .

# 2. Ejecutar con acceso a GPU
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace --name transformer_gpu p1_rayosx_transformer:gpu

# 3. Abrir Jupyter -> http://localhost:8888
# 4. Para detener:
docker stop transformer_gpu
```

Dentro del notebook, verifica GPU con:
```python
import torch
torch.cuda.is_available()  # debe devolver True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
</details>

### Perfilador de apoyo

```python
# src/profiling.py
import time, torch, math
import torch.nn as nn
from contextlib import nullcontext

def count_params(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def benchmark(modelo, seq_len=256, batch_size=16, steps=50, use_amp=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device).eval()
    vocab_size = getattr(modelo, "vocab_size", 30522)  # ajusta si usas otro tokenizer
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attn = torch.ones_like(ids)

    scaler_ctx = torch.cuda.amp.autocast if (use_amp and device=="cuda") else nullcontext
    torch.cuda.reset_peak_memory_stats(device) if device=="cuda" else None

    # warmup
    for _ in range(5):
        with scaler_ctx():
            _ = modelo(ids, attention_mask=attn)

    t0 = time.perf_counter()
    for _ in range(steps):
        with scaler_ctx():
            _ = modelo(ids, attention_mask=attn)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tokens = steps * batch_size * seq_len
    tps = tokens / elapsed
    mem = torch.cuda.max_memory_reserved(device)/1e9 if device=="cuda" else None

    return {
        "device": device,
        "params_m": round(count_params(modelo)/1e6, 3),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "steps": steps,
        "elapsed_s": round(elapsed, 3),
        "tokens": tokens,
        "tokens_per_s": round(tps, 1),
        "peak_mem_gb": round(mem, 3) if mem is not None else None,
        "amp": bool(use_amp),
    }
```

#### Cómo usarlo dentro del notebook

```python
from src.profiling import benchmark
# asumiendo que tu clase se llama TinyEncoderClassifier y que fijaste vocab_size
modelo.vocab_size = tok.vocab_size  # para el perfilador
res = benchmark(modelo, seq_len=256, batch_size=16, steps=100, use_amp=True)
res
```

#### Target Makefile rápido (opcional)

Añade esto al `Makefile` del P1 para ejecutar el perfil en la imagen Docker:

```make
profile:
\tdocker run --rm -it -v $$(pwd):/workspace --entrypoint python $(IMAGE) - <<'PY'
from src.profiling import benchmark
# IMPORTA tu modelo desde src o notebook exportado
from notebooks.utils_export import modelo  # ajusta si lo tienes en otro lugar
print(benchmark(modelo, seq_len=256, batch_size=16, steps=100, use_amp=True))
PY
```

#### Tips de interpretación

* **tokens_per_s**: si baja mucho al subir `seq_len`, recorta `max_length` (la atención es O(L^2)).
* **peak_mem_gb (GPU)**: si > VRAM, baja `batch_size` o `seq_len`.
* **amp=True** (mixed precision) suele mejorar velocidad y bajar memoria en GPU.

