### Proyecto 4 - BERT: Fine-tuning vs LoRA vs Adapters

**Tema:** PEFT (Parameter-Efficient Fine-Tuning) y eficiencia.

#### Objetivo

Demostrar **cuantitativa y visualmente** por qué en producción casi nadie hace full fine-tuning de BERT.  Se debe compararar **tres estrategias** en la misma tarea de clasificación:  

- **Full Fine-Tuning** (todos los parámetros entrenables).  
- **LoRA** (r=8 y r=16) + **QLoRA** (4-bit).  
- **Adapters** (Hugging Face o Pfeiffer/Houston).  

El objetivo es responder:  
- ¿Cuánto pierdo en calidad al usar PEFT?  
- ¿Cuánto gano en memoria, velocidad y capacidad de multitarea?

#### Dataset
- **Recomendado**: SST-2 (binario, sentimiento de frases cortas, ~67k ejemplos). Más rápido y estable.  
- **Alternativa**: AG News (4 clases, tópicos, ~120k ejemplos). Más realista para producción.  
- Usa `datasets.load_dataset("glue", "sst2")` o `"Agnews"` desde Hugging Face.

#### Entregables
- **Notebook principal** con:  
  - Pipeline completo y **reproducible** (seed 42 en todo: torch, numpy, random, huggingface).  
  - 5 corridas:  
    1. Full FT  
    2. LoRA r=8  
    3. LoRA r=16  
    4. QLoRA (4-bit, r=16)  
    5. Adapters (botella 64, reducción 8)  
  - Tabla automática de **parámetros entrenables** y **% del total**.  
- **Carpeta `figures/`** con:  
  - Curvas de **loss y F1 macro** por época (las 5 en una sola gráfica).  
  - Barra de **memoria pico** (GPU VRAM) durante entrenamiento.  
  - Barra de **tiempo por época**.  
  - Tabla resumen en PNG/Markdown.  
- **Carpeta `metrics/`** con `comparison.json` ejemplo:  
  ```json
  {
    "full_ft": {"trainable_params": 109486849, "f1_macro": 0.932, "peak_vram_gb": 11.8, "epoch_time_s": 185},
    "lora_r8": {"trainable_params": 294912, "f1_macro": 0.928, "peak_vram_gb": 4.2, "epoch_time_s": 92},
    "qlora": {"trainable_params": 294912, "f1_macro": 0.926, "peak_vram_gb": 2.9, "epoch_time_s": 108}
  }
  ```

#### Métricas
- **F1 macro** (principal, más robusto que accuracy).  
- **# parámetros entrenables** (exacto, no aproximado).  
- **Memoria pico** (usa `torch.cuda.max_memory_allocated() / 1e9`).  
- **Tiempo por época** (promedio de las 3 épocas).  
- Se espera:  
  - Full FT: F1 > 0.93  
  - LoRA/QLoRA: F1 > 0.92 (pérdida < 1 %)  
  - Adapters: F1 > 0.91  
  - QLoRA: < 3 GB VRAM en una RTX 3060 12GB.

#### Pasos (desglosados con detalle)
1. **Splits y seed fija**  
   - Usa `datasets` con `load_dataset(..., split=...)`.  
   - Train/val/test fijos.  
   - `set_seed(42)` en todos lados (incluye `transformers.set_seed(42)`).  
   - Tokenizador: `AutoTokenizer.from_pretrained("bert-base-uncased")`.

2. **Implementar LoRA y Adapters**  
   - **Full FT**: `BertForSequenceClassification` normal.  
   - **LoRA**: usa `peft` library -> `LoraConfig(r=8/16, lora_alpha=32, target_modules=["query", "value"])`  
   - **QLoRA**:  
     - `bitsandbytes` 4-bit  
     - `load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.bfloat16`  
   - **Adapters**: `AdapterConfig(type="houston", reduction_factor=8)` o Pfeiffer.  
   - Todos con **misma cabeza de clasificación** (nn.Linear) entrenable.

3. **Evaluar eficiencia y calidad**  
   - Entrena **3 épocas** con batch size 32 (acumula si es necesario).  
   - Learning rate: 2e-5 (full) / 1e-4 (PEFT).  
   - Warmup 10 % steps.  
   - Evalúa en test **solo el mejor modelo** por val F1.  
   - Mide memoria con:  
     ```python
     torch.cuda.reset_peak_memory_stats()
     # entrenamiento
     peak = torch.cuda.max_memory_allocated() / 1e9
     ```

#### Video
**Guion recomendado**:  
1. **Introducción** : "¿Por qué nadie fine-tunea todo BERT en producción? Hoy lo vemos con números reales."  
2. **Demostración en vivo**:  
   - Muestra la tabla comparativa en tiempo real.  
   - Cambia entre modelos y muestra predicciones en 5 frases ambiguas.  
   - Zoom en VRAM: "Full FT come 11.8 GB… QLoRA solo 2.9 GB."  
3. **Gráficas**: curvas de F1, barras de tiempo/memoria.  
4. **Takeaways de producción**:  
   - "LoRA r=16 pierde solo 0.4 % F1 pero entrena 2× más rápido y cabe en 4 GB."  
   - "QLoRA permite fine-tunear BERT en una laptop con 8 GB VRAM."  
   - "Adapters son buenos para multitarea (puedes tener 50 tareas en un solo modelo)."  
   - "Full FT solo justifica si tienes GPUs gigantes y necesitas el último 0.5 %."  
   - Cierre épico: "En 2025, quien hace full fine-tuning..."

### Ejecutar con Docker
> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):  
   ```bash
   make build
   ```
   -> Crea la imagen `p4_bert_vs_lora_vs_adapters`.

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
- El volumen `-v $PWD:/workspace` salva **todo** localmente.  
- `requirements.txt` obligatorio:  
  ```
  torch>=2.1
  transformers>=4.38
  datasets
  peft>=0.8
  bitsandbytes>=0.43
  accelerate
  sentencepiece
  jupyterlab
  matplotlib
  seaborn
  pandas
  tqdm
  ```
- Usa `accelerate config` dentro del contenedor si quieres mixed precision automático.

<details><summary>Ejemplo (opcional) con GPU CUDA - más explicado</summary>

**Por qué GPU es imprescindible aquí**:  
- Full FT en batch 32 -> 11-12 GB VRAM.  
- QLoRA -> cabe en 6 GB.  
- Sin GPU: full FT tardaría ~3 horas vs 15 min con GPU.

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
docker build -t p4_bert_vs_lora_vs_adapters:gpu -f Dockerfile.gpu .

# 2. Ejecutar con GPU
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace --name bert_peft_gpu p4_bert_vs_lora_vs_adapters:gpu

# 3. Abrir http://localhost:8888
# 4. Detener
docker stop bert_peft_gpu
```

Verifica todo en notebook:
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))
print("VRAM total:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```
</details>

