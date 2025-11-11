### Proyecto 5 - Lenguajes causales: PPL y decodificación

**Tema:** PPL/cross-entropy y calidad generativa.

#### Objetivo
Demostrar que **baja PPL no siempre = buena generación**.  
Se entrenará un **modelo causal pequeño** (GPT-style) y se analizará exhaustivamente cómo diferentes estrategias de decodificación afectan la **calidad subjetiva** aunque la perplexity sea idéntica.  

Se explorará:

- Greedy  
- Beam search (k=4)  
- Top-k (k=40)  
- Top-p (nucleus, p=0.9/0.95)  
- Temperatura (T=0.7/1.0/1.2)  
- Combinaciones peligrosas (greedy + T>1 -> repetición infinita)

#### Dataset
- **Recomendado**: WikiText-2 (raw, ~2M tokens). Perfecto por ser limpio y real.  
  - Usa `datasets.load_dataset("wikipedia", "20220301.en")` y toma un subset de 500k tokens para entrenamiento rápido.  
- **Alternativa divertida**: Cuentos de los Hermanos Grimm o poesía en español (para ver repetición de rimas).  
- Tokenización: **Byte-level BPE** (GPT-2 tokenizer) o entrena uno propio con 16k vocab.

#### Entregables
- **Notebook principal** con:  
  - Entrenamiento desde cero de un GPT-2 pequeño (6 capas, d_model=512, 8 heads).  
  - Cálculo preciso de **cross-entropy y PPL** en validación (sin padding trick).  
  - **Grilla interactiva** de generación: 10 prompts × 8 configuraciones de decodificación.  
  - Métricas automáticas por muestra generada.  
- **Carpeta `samples/`** con:  
  - `generations_grid.csv` (prompt + 8 salidas + métricas).  
  - 5 ejemplos **destacados** en Markdown (el peor, el mejor, el más repetitivo, el más creativo, el más "humano").  
- **Carpeta `figures/`** con:  
  - Scatter plot: **PPL vs repetición** (color por método).  
  - Boxplot de **longitud media** y **type/token ratio**.  
  - Heatmap de **frecuencia de bigrams repetidos**.  
- **Carpeta `metrics/`** con `decoding_analysis.json` ejemplo:  
  ```json
  {
    "greedy": {"ppl": 18.4, "avg_len": 42, "repeat_rate": 0.32, "ttr": 0.61},
    "top_p_0.92": {"ppl": 18.4, "avg_len": 78, "repeat_rate": 0.03, "ttr": 0.88},
    "temp_1.2": {"ppl": 18.4, "avg_len": 120, "repeat_rate": 0.01, "ttr": 0.91}
  }
  ```

#### Métricas
- **PPL** (perplexity = exp(cross_entropy)).  
- **Longitud media** de generación (max 128 tokens).  
- **Repetición**: % de n-gramas (2-4) que aparecen >2 veces.  
- **Type/Token Ratio** (diversidad léxica).  
- **Human-like score** (opcional): tú puntúas 10 muestras de 1-5.  
- Resultado esperado:  
  - Greedy: PPL baja, pero repetición alta ("the the the").  
  - Top-p 0.92 + T=1.0: PPL igual, pero calidad **mucho** mejor.  
  - Temperatura >1.2: explota en diversidad, pero coherencia cae.

#### Pasos (desglosados con detalle)
1. **Calcular CE/PPL en validación**  
   - Usa `model.eval()` + `torch.no_grad()`.  
   - Ignora padding y tokens de control.  
   - PPL final < 25 es excelente para modelo pequeño.

2. **Experimentos de decodificación**  
   - Implementa **todas** las estrategias desde cero (¡no uses `generate` de transformers!).  
   - Función `generate(prompt, strategy, **kwargs)` que devuelva texto + logprobs.  
   - 10 prompts diversos:  
     - Factual ("The capital of France is")  
     - Creativo ("In a hole in the ground there lived")  
     - Abierto ("Once upon a time")  
     - Técnico ("The transformer architecture was introduced in")  
   - Genera con:  
     - greedy  
     - beam=4  
     - top_k=40  
     - top_p=0.9 / 0.95  
     - temperature=0.7 / 1.0 / 1.2  
     - top_p=0.92 + T=0.8 (ganadora típica)

3. **Analizar correlación PPL vs calidad**  
   - Gráfica clave: PPL en eje X (todos ~igual), calidad subjetiva en Y.  
   - Conclusión esperada: "PPL mide fluidez, no creatividad ni coherencia".  
   - Demuestra que **top-p sampling rompe la maldición del greedy** sin empeorar PPL.

#### Video
**Guion recomendado (en vivo, súper impactante)**:  
1. **Intro**: "¿Por qué ChatGPT no repite 'the the the'? Hoy muestro que la PPL no lo es todo."  
2. **Demo en vivo**:  
   - Abre un widget interactivo (usa `ipywidgets` o `gradio` dentro del notebook).  
   - Selecciona prompt -> cambia estrategia en dropdown -> genera al instante.  
   - Ejemplos épicos:  
     - Greedy: "The cat sat on the mat mat mat mat..."  
     - Top-p 0.92: "The cat sat on the windowsill, watching raindrops race down the glass..."  
   - Muestra métricas en tiempo real debajo de cada texto.  
3. **Análisis** : scatter plot + boxplots.  
4. **Takeaways**:  
   - "Nunca uses greedy en producción."  
   - "Top-p 0.92 es el estándar de oro por una razón."  
   - Cierre: "Ahora sabes por qué los LLMs suenan humanos... y tú también puedes."

### Ejecutar con Docker
> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):  
   ```bash
   make build
   ```
   -> Crea la imagen `p5_causallm_ppl_and_decoding`.

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
- `requirements.txt` debe incluir:  
  ```
  torch>=2.1
  transformers
  datasets
  tokenizers
  matplotlib
  seaborn
  pandas
  numpy
  tqdm
  jupyterlab
  ipywidgets
  gradio  # opcional pero recomendado para demo interactiva
  ```
- Entrenamiento: ~30-45 min en GPU (6 épocas, batch 32, seq_len 256).

<details><summary>Ejemplo (opcional) con GPU CUDA - más explicado</summary>

**Por qué GPU es clave aquí**:  
- Entrenamiento en CPU: >4 horas.  
- Con GPU (RTX 3060+): 30-40 min total.  
- Generación interactiva fluida.

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
docker build -t p5_causallm_ppl_and_decoding:gpu -f Dockerfile.gpu .

# 2. Ejecutar con GPU
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace --name causal_gpu p5_causallm_ppl_and_decoding:gpu

# 3. Abrir http://localhost:8888
# 4. Detener
docker stop causal_gpu
```

Verifica GPU:
```python
import torch
print(torch.cuda.is_available())  # True
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```
</details>
