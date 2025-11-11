### Proyecto 6 - Turbo-Inferencia: KV-Cache y Cuantización

**Tema:** Rendimiento en tiempo de prueba.

#### Objetivo

Demostrar **cuánto se puede acelerar la inferencia** de un LLM grande sin sacrificar calidad, combinando dos técnicas clave:  
- **KV-Cache** (prefill vs decoding)  
- **Cuantización** (FP16 -> BF16 -> INT8 -> 4-bit)  

Se medirá la **latencia real**, **tokens/s**, **memoria VRAM** y **PPL** en contextos largos (128 -> 2048 tokens) para responder:  
- ¿Cuánto gana el KV-cache en fase de decoding?  
- ¿Vale la pena cuantizar a 4-bit en producción?  
- ¿Dónde está el sweet spot entre velocidad y calidad?

#### Dataset
- **Prompts sintéticos** generados con plantillas:  
  - Conversacionales ("User: Explícame... Assistant:")  
  - Código ("def quicksort...")  
  - Resúmenes largos ("Summarize the following article...")  
- Longitudes: **128, 256, 512, 1024, 2048 tokens**.  
- 50 prompts por longitud -> estadísticas robustas.  
- Modelo base: **Llama-3-8B-Instruct** o **Mistral-7B-Instruct-v0.3** (disponibles en Hugging Face).

#### Entregables
- **Notebook principal** con:  
  - Benchmark automático (warmup + 10 corridas por config).  
  - Tabla comparativa **clara y colorida** (FP16, BF16, INT8, 4-bit).  
  - Gráficos interactivos (Plotly recomendado).  
- **Carpeta `benchmarks/`** con:  
  - `results.csv` (todas las corridas).  
  - `summary_table.png` + `latency_vs_context.png`.  
  - `memory_footprint.png` (VRAM por longitud).  
- **Carpeta `figures/`** con:  
  - **Latencia prefill vs decoding** (log scale).  
  - **Tokens/s vs longitud** (curvas por cuantización).  
  - **PPL degradation** (debe ser < 5 % en 4-bit).  
- **Carpeta `metrics/`** con `final_report.json` ejemplo:  
  ```json
  {
    "fp16": {"tokens_s_2048": 42.1, "vram_gb": 16.8, "prefill_ms": 890, "ppl": 12.4},
    "int8": {"tokens_s_2048": 78.3, "vram_gb": 9.2, "prefill_ms": 720, "ppl": 12.6},
    "4bit": {"tokens_s_2048": 112.5, "vram_gb": 5.9, "prefill_ms": 680, "ppl": 13.1}
  }
  ```

#### Métricas
- **Tokens/s** (decoding phase, promedio).  
- **Latencia total** (prefill + first token + decoding).  
- **Memoria pico** (VRAM en GB).  
- **PPL** en WikiText-2 val (para verificar degradación).  
- **Resultados esperados (RTX 4090 / A100 40GB)**:  
  - FP16: ~40-50 tokens/s a 2048 tokens  
  - INT8: ~75-90 tokens/s  
  - 4-bit (GPTQ/AWQ): **110-130 tokens/s**  
  - KV-cache desactivado: decoding cae a < 5 tokens/s

#### Pasos (desglosados con detalle)
1. **Script con warmup y seeds**  
   - Usa `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`.  
   - **Warmup obligatorio**: 5 corridas desechadas para estabilizar CUDA.  
   - Mide con `time.perf_counter()` y `torch.cuda.Event`.

2. **Activar/desactivar KV-cache, variar longitud**  
   - Usa `use_cache=True/False` en `modelo.generate()`.  
   - Separa claramente:  
     - **Prefill time** (hasta first token).  
     - **Decoding time** (tokens 2..N).  
   - Demuestra que **KV-cache es mágico** en decoding: latencia O(1) por token.

3. **Cargar pesos cuantizados y comparar**  
   - Configuraciones:  
     - FP16 (base)  
     - BF16 (si tu GPU soporta)  
     - INT8 (weight-only, `bitsandbytes`)  
     - 4-bit (GPTQ o AWQ, `AutoGPTQ` o `exllamav2`)  
   - Usa `device_map="auto"` + `torch_dtype=torch.float16`.  
   - Verifica PPL con `evaluate` library o script propio.

#### Video
**Guion recomendado (demo en vivo, impactante)**:  
1. **Introducción** : "¿Cómo sirve un LLM a 120 tokens/s en una sola GPU? Hoy presentamos los dos trucos que lo hacen posible."  
2. **Demostración benchmark en vivo**:  
   - Abre dashboard (Plotly Dash o Gradio).  
   - Selecciona longitud -> cuantización -> corre benchmark en tiempo real.  
   - Muestra:  
     - "Sin KV-cache: 4 tokens/s a 2048 tokens... ¡imposible servir!"  
     - "Con KV-cache + 4-bit: 118 tokens/s, solo 5.9 GB VRAM."  
   - Gráfica de latencia vs contexto: curva plana con KV-cache.  
3. **Memoria y PPL**:  
   - "4-bit cabe en una RTX 3060 12GB, pierde solo 0.7 PPL."  
4. **Conclusiones prácticas** :  
   - "KV-cache es obligatorio para producción."  
   - "INT8 es el sweet spot para latencia sin complicaciones."  
   - "4-bit (AWQ) es el futuro: velocidad de vLLM con 40 % menos VRAM."  
   - "Nunca desactives KV-cache a menos que quieras morir de latencia."  
   - Cierre épico: "Ahora sabes por qué ... responde en <1 segundo aunque tenga 2048 tokens de contexto."

### Ejecutar con Docker
> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):  
   ```bash
   make build
   ```
   -> Crea la imagen `p6_inference_kvcache_quantization`.

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
- El volumen `-v $PWD:/workspace` guarda **todo** localmente.  
- `requirements.txt` obligatorio:  
  ```
  torch>=2.3
  transformers>=4.40
  accelerate
  bitsandbytes>=0.43
  auto-gptq
  optimum
  autoawq
  datasets
  evaluate
  plotly
  pandas
  numpy
  jupyterlab
  gradio  # para dashboard opcional
  ```
- Usa `torch.compile(model)` si tienes PyTorch 2.3+ -> +15 % tokens/s gratis.

<details><summary>Ejemplo (opcional) con GPU CUDA - más explicado</summary>

**Por qué GPU es 100 % obligatorio aquí**:  
- Llama-3-8B en FP16: 16 GB VRAM mínimo.  
- Sin GPU: imposible cargar el modelo.  
- Con 4-bit: cabe en RTX 3060 12GB.

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
docker build -t p6_inference_kvcache_quantization:gpu -f Dockerfile.gpu .

# 2. Ejecutar con GPU
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace --name turbo_gpu p6_inference_kvcache_quantization:gpu

# 3. Abrir http://localhost:8888
# 4. Detener
docker stop turbo_gpu
```

Verifica todo:
```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM total:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```
</details>
