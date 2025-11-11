### Proyecto 2 - El tiempo en el espacio: Positional Encodings

**Tema:** Sinusoidal, learned, RoPE, ALiBi, extrapolación a largo contexto.

#### Objetivo

Demostrar **cuantitativa y visualmente** cómo diferentes codificaciones posicionales afectan la capacidad del modelo para **extrapolación en longitud**.  Se debe entrenar **todos los modelos en secuencias de máximo 256 tokens** y luego evaluar en **512 y 1024 tokens** (el doble y el cuádruple).  
El objetivo no es solo ver quién gana, sino **explicar por qué** una codificación falla o triunfa desde la teoría (rotaciones, bias lineal, dependencia absoluta vs relativa, etc.).

#### Dataset
- **Tarea sintética principal** (copy/duplicate):  
  - Genera secuencias aleatorias de enteros (vocabulario pequeño, ejemplo, 0-99).  
  - Tarea **copy**: entrada = `[a1, a2, ..., an, SEP, 0, 0, ..., 0]`, salida = `[a1, a2, ..., an]`.  
  - Tarea **duplicate**: salida = `[a1, ..., an, a1, ..., an]`.  
  - Ventaja: el éxito depende 100 % de la posición -> degradación muy visible al extrapolar.  
- **Tarea real complementaria**: IMDb con reseñas largas.  
  - Trunca a 256 durante entrenamiento, pero evalúa en reseñas completas (hasta ~1000 tokens).  
  - Permite ver el impacto en un problema real.

#### Entregables
- **Notebook principal** con:  
  - Implementación limpia de las **cuatro** codificaciones posicionales (sin usar bibliotecas externas que las oculten).  
  - Mismo bloque Transformer (reusa el del Proyecto 1) + misma semilla para que la única variable sea la PE.  
  - Tablas comparativas automáticas.  
- **Carpeta `figures/`** con:  
  - Gráfica de **accuracy vs longitud** (256, 512, 1024) para la tarea sintética (una línea por cada PE).  
  - Gráfica de **pérdida por longitud** en IMDb.  
  - Heatmaps de **norma de gradientes** por capa a longitud 1024 (para ver explosión/desvanecimiento).  
  - Curvas de pérdida de entrenamiento (para verificar que todas convergen igual a 256).  
- **Carpeta `metrics/`** con `results.json` que incluya:  
  ```json
  {
    "synthetic": {
      "256": {"sinusoidal": 0.99, "learned": 0.98, "rope": 0.99, "alibi": 0.97},
      "512": {"sinusoidal": 0.12, "learned": 0.08, "rope": 0.98, "alibi": 0.95},
      "1024": {"sinusoidal": 0.05, "learned": 0.03, "rope": 0.96, "alibi": 0.90}
    },
    "imdb": {"acc_256": X.XX, "acc_512": X.XX, "acc_1024": X.XX}
  }
  ```

#### Métricas
- **Tarea sintética**: Accuracy exacta (debe ser >98 % a 256, idealmente 100 %).  
- **IMDb**: Accuracy y F1 (macro).  
- **Estabilidad numérica**: Norma media de gradientes en la última capa a longitud 1024, pérdida máxima alcanzada (NaN = 0 puntos).  
- Se espera: **RoPE y ALiBi** mantengan >90 % en 1024; **sinusoidal y learned** colapsen a <20 %.

#### Pasos (desglosados con detalle)
1. **Inyectar cada PE en el mismo modelo**  
   - Crea una clase base `PositionalEncoding` con método `forward(self, x, seq_len)` que devuelva el mismo tamaño que `x`.  
   - Implementa cuatro subclases:  
     - `SinusoidalPE` (fijo, sin parámetros).  
     - `LearnedPE` (nn.Embedding trainable, max 256).  
     - `RoPE` (aplicado en las matrices Q y K antes del dot-product. Usa frecuencia base 10000).  
     - `ALiBi` (no suma, sino **bias** triangular restado a los attn_scores, pendiente negativa).  
   - Usa exactamente el mismo bloque Transformer del Proyecto 1 (mismas dimensiones, mismas cabeceras).

2. **Entrenar corto a 256 tokens, probar a 512/1k**  
   - Entrena **4 modelos idénticos** (misma semilla) diferenciados solo por la PE.  
   - 3-5 épocas máximo; la tarea sintética converge rapidísimo.  
   - Durante evaluación, **desactiva el truncado** y pasa secuencias completas de 512 y 1024.  
   - Para RoPE no hay límite teórico, para ALiBi tampoco, para learned y sinusoidal **no** extiendas el embedding más allá de 256.

3. **Analizar degradación y justificación técnica**  
   - Explica en celdas markdown:  
     - Sinusoidal: falla porque las frecuencias no están entrenadas para longitudes mayores -> patrones fuera de rango.  
     - Learned: OOV posicional puro -> atención se vuelve ruido.  
     - RoPE: preserva distancias relativas mediante rotaciones -> extrapolación perfecta.  
     - ALiBi: bias lineal penaliza distancias grandes de forma aprendible -> muy robusto.  
   - Incluye una tabla resumen con pros/contras y complejidad computacional.

#### Video 
**Guion recomendado**:  
1. **Introducción**: "¿Qué pasa cuando le pides a un Transformer que lea el doble de largo de lo que entrenó? Hoy comparamos 4 formas de decirle 'dónde está cada palabra'."  
2. **Demo tarea sintética**:  
   - Muestra predicciones a 256 (todos perfectos) -> 512 (sinusoidal y learned fallan estrepitosamente) -> 1024 (solo RoPE y ALiBi sobreviven).  
   - Zoom en los mapas de atención: learned pone atención uniforme, RoPE mantiene picos claros.  
3. **IMDb real** : gráfica de accuracy vs longitud.  
4. **Estabilidad** : gráficos de gradientes y explicación de NaNs en **learned**.  
5. **Conclusiones** (1 min):  
   - "RoPE es el rey de la extrapolación absoluta."  
   - "ALiBi es sorprendentemente bueno y más barato."  
   - "Sinusoidal y learned solo sirven hasta la longitud entrenada."  
   - Cierra con frase épica: "El tiempo (posicional) sí importa... y ahora sabemos cuál resiste el paso del tiempo."

### Ejecutar con Docker
> Requisitos: Docker Desktop o Docker Engine reciente.

1. **Construir imagen** (una sola vez o cuando cambie `requirements.txt`):  
   ```bash
   make build
   ```
   -> Crea la imagen `p2_positional_encodings`.

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
- El volumen `-v $PWD:/workspace` garantiza que **nunca pierdes notebooks ni figuras**.  
- `requirements.txt` debe incluir: `torch`, `torchtext`, `matplotlib`, `seaborn`, `numpy`, `tqdm`, `einops` (útil para RoPE), `jupyterlab`.  
- Usa `torch.manual_seed(42)` y `torch.cuda.manual_seed_all(42)` en todos los experimentos para reproducibilidad perfecta.

<details><summary>Ejemplo (opcional) con GPU CUDA - más explicado</summary>

**Ventaja GPU**:  
- Tarea sintética a 1024 tokens con batch 32 -> < 30 segundos por época.  
- Sin GPU puede tardar 10-15 min por experimento.

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
docker build -t p2_positional_encodings:gpu -f Dockerfile.gpu .

# 2. Ejecutar con GPU
docker run --rm -d --gpus all -p 8888:8888 -v $PWD:/workspace --name pe_gpu p2_positional_encodings:gpu

# 3. Abrir http://localhost:8888
# 4. Detener
docker stop pe_gpu
```

Dentro del notebook verifica:

```python
import torch
print(torch.cuda.is_available())  # True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
</details>

